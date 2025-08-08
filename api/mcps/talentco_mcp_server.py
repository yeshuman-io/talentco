import os
import asyncio
import json

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "talentco.settings")

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

import django  # noqa: E402

django.setup()

from mcp.server.fastmcp import FastMCP  # type: ignore
from asgiref.sync import sync_to_async  # type: ignore

# Import services and models after Django setup
from apps.applications.services import ApplicationService  # noqa: E402
from apps.evaluations.services import EvaluationService  # noqa: E402
from apps.profiles.models import Profile  # noqa: E402
from apps.opportunities.models import Opportunity  # noqa: E402
from apps.applications.models import Application  # noqa: E402
from django.db.models import Count  # noqa: E402
from graphs import agent_tools  # noqa: E402
from graphs.agent import create_employer_agent, create_candidate_agent, create_general_agent  # noqa: E402

# JSON envelope helpers (mirrors agent_tools)
def ok(data: dict, message: str = "") -> str:
    return json.dumps({"status": "ok", "data": data, "message": message})

def err(code: str, message: str, *, hints=None, fields=None, suggestions=None) -> str:
    return json.dumps({
        "status": "error",
        "code": code,
        "message": message,
        "hints": hints or [],
        "fields": fields or [],
        "suggestions": suggestions or {},
    })


server = FastMCP("talentco-mcp")
app_service = ApplicationService()
eval_service = EvaluationService()


@server.tool()
async def list_profiles(limit: int = 20, search: str | None = None) -> str:
    def q():
        qs = Profile.objects.all().order_by("-created_at")
        if search:
            qs = qs.filter(first_name__icontains=search) | qs.filter(last_name__icontains=search) | qs.filter(email__icontains=search)
        data = []
        for p in qs[:limit]:
            data.append({
                "id": str(p.id),
                "first_name": p.first_name,
                "last_name": p.last_name,
                "email": p.email,
            })
        return data
    rows = await sync_to_async(q)()
    return ok({"profiles": rows}, message=f"{len(rows)} profiles")


@server.tool()
async def list_opportunities(limit: int = 20, search: str | None = None) -> str:
    def q():
        qs = Opportunity.objects.select_related("organisation").annotate(
            questions_count=Count("questions"), skills_count=Count("opportunity_skills")
        ).order_by("-created_at")
        if search:
            qs = qs.filter(title__icontains=search) | qs.filter(organisation__name__icontains=search)
        return [
            {
                "id": str(o.id),
                "title": o.title,
                "organisation": o.organisation.name,
                "questions_count": o.questions_count,
                "skills_count": o.skills_count,
            }
            for o in qs[:limit]
        ]
    rows = await sync_to_async(q)()
    return ok({"opportunities": rows}, message=f"{len(rows)} opportunities")


@server.tool()
async def list_applications_for_opportunity(opportunity_id: str, status: str | None = None, limit: int = 50) -> str:
    def q():
        qs = Application.objects.select_related(
            "profile", "current_stage_instance__stage_template"
        ).filter(opportunity_id=opportunity_id).order_by("-applied_at")
        if status:
            qs = qs.filter(status=status)
        data = []
        for a in qs[:limit]:
            data.append({
                "id": str(a.id),
                "profile": f"{a.profile.first_name} {a.profile.last_name}",
                "status": a.status,
                "current_stage": a.current_stage_instance.stage_template.slug if a.current_stage_instance else None,
                "applied_at": a.applied_at.isoformat(),
            })
        return data
    rows = await sync_to_async(q)()
    return ok({"applications": rows}, message=f"{len(rows)} applications")


@server.tool()
async def get_application(application_id: str) -> str:
    def q():
        a = Application.objects.select_related(
            "profile", "opportunity__organisation", "current_stage_instance__stage_template"
        ).prefetch_related("answers__question", "interviews").get(id=application_id)
        return {
            "id": str(a.id),
            "status": a.status,
            "profile": {
                "id": str(a.profile.id),
                "name": f"{a.profile.first_name} {a.profile.last_name}",
                "email": a.profile.email,
            },
            "opportunity": {
                "id": str(a.opportunity.id),
                "title": a.opportunity.title,
                "organisation": a.opportunity.organisation.name,
            },
            "current_stage": a.current_stage_instance.stage_template.slug if a.current_stage_instance else None,
            "answers": [
                {
                    "id": str(ans.id),
                    "question_id": str(ans.question_id),
                    "question_text": ans.question.question_text,
                    "type": ans.question.question_type,
                    "is_required": ans.question.is_required,
                    "is_disqualifying": ans.is_disqualifying,
                } for ans in a.answers.all()
            ],
            "interviews": [
                {
                    "id": str(iv.id),
                    "round_name": iv.round_name,
                    "start": iv.scheduled_start.isoformat(),
                    "end": iv.scheduled_end.isoformat(),
                    "location_type": iv.location_type,
                    "outcome": iv.outcome,
                } for iv in a.interviews.all()
            ]
        }
    try:
        data = await sync_to_async(q)()
        return ok(data, message="Application details")
    except Exception as e:
        return err("not_found", f"{e}")


@server.tool()
async def seed_default_stages() -> str:
    stages = await sync_to_async(app_service.seed_default_stages)()
    return ok({"slugs": [s.slug for s in stages]}, message="Stages ensured")


@server.tool()
async def list_stages() -> str:
    stages = await sync_to_async(app_service.list_stages)()
    data = [{"id": str(s.id), "slug": s.slug, "name": s.name, "order": s.order, "is_terminal": s.is_terminal} for s in stages]
    return ok({"stages": data}, message=f"{len(data)} stages")


@server.tool()
async def upsert_stage(payload: dict) -> str:
    try:
        st = await sync_to_async(app_service.upsert_stage_template)(payload)
        return ok({"id": str(st.id), "slug": st.slug, "name": st.name, "order": st.order, "is_terminal": st.is_terminal}, message="Stage saved")
    except Exception as e:
        return err("upsert_failed", str(e))


@server.tool()
async def delete_stage(identifier: str) -> str:
    try:
        count = await sync_to_async(app_service.delete_stage_template)(identifier)
        return ok({"deleted": count}, message="Stage deleted")
    except Exception as e:
        return err("delete_failed", str(e))


@server.tool()
async def create_application_for_profile(profile_id: str, opportunity_id: str, source: str = "direct", answers: list | None = None) -> str:
    try:
        result = await sync_to_async(app_service.create_application)(profile_id, opportunity_id, source, answers)
    except Profile.DoesNotExist:
        return err("not_found", f"Profile {profile_id} not found", suggestions={"try": "list_profiles"})
    except Opportunity.DoesNotExist:
        return err("not_found", f"Opportunity {opportunity_id} not found", suggestions={"try": "list_opportunities"})
    except Exception as e:
        return err("create_failed", str(e))
    app = result.application
    return ok({"application_id": str(app.id), "status": app.status, "result": "created" if result.created else "already_exists"})


@server.tool()
async def change_application_stage(application_id: str, stage_slug: str) -> str:
    try:
        app = await sync_to_async(app_service.change_stage)(application_id, stage_slug)
        return ok({"application_id": str(app.id), "status": app.status, "current_stage": app.current_stage_instance.stage_template.slug if app.current_stage_instance else None}, message="Stage changed")
    except Exception as e:
        slugs = [s.slug for s in await sync_to_async(app_service.list_stages)()]
        return err("change_failed", str(e), hints=["Seed stages first"], suggestions={"available_slugs": slugs})


@server.tool()
async def decision_application(application_id: str, status: str, reason: str | None = None) -> str:
    try:
        app = await sync_to_async(app_service.record_decision)(application_id, status, reason)
        return ok({"application_id": str(app.id), "status": app.status}, message="Decision recorded")
    except AssertionError as e:
        return err(
            "invalid_status",
            str(e) or "Invalid decision status",
            hints=["Use one of: hired, rejected, offer"],
            suggestions={"allowed_values": {"status": ["hired", "rejected", "offer"]}},
        )
    except Exception as e:
        return err("decision_failed", str(e))


@server.tool()
async def upsert_screening_question(opportunity_id: str, question: dict | None = None, **flat) -> str:
    # Accept nested or flat
    q = dict(question) if isinstance(question, dict) else {}
    q.update({k: v for k, v in flat.items() if v is not None})
    # Normalize type aliases
    if "question_type" in q and isinstance(q["question_type"], str):
        qt = q["question_type"].strip().lower().replace("-", "_")
        alias = {"single": "single_choice", "multi": "multi_choice", "numeric": "number", "int": "number"}
        q["question_type"] = alias.get(qt, qt)
    # Validate
    missing = []
    if not q.get("id") and not q.get("question_text"):
        missing.append("question_text")
    if not q.get("id") and not q.get("question_type"):
        missing.append("question_type")
    if missing:
        return err("missing_field", "Missing required fields", fields=missing, suggestions={"allowed_values": {"question_type": ["text","boolean","single_choice","multi_choice","number"]}})
    try:
        saved = await sync_to_async(app_service.upsert_screening_question)(opportunity_id, q)
        return ok({"id": str(saved.id), "question_text": saved.question_text, "question_type": saved.question_type, "is_required": saved.is_required, "order": saved.order, "config": saved.config}, message="Question saved")
    except Exception as e:
        return err("upsert_failed", str(e))


@server.tool()
async def delete_screening_question(question_id: str) -> str:
    try:
        await sync_to_async(app_service.delete_screening_question)(question_id)
        return ok({"question_id": question_id}, message="Question deleted")
    except Exception as e:
        return err("delete_failed", str(e))


@server.tool()
async def list_screening_questions(opportunity_id: str) -> str:
    from apps.applications.models import OpportunityQuestion
    def q():
        data = []
        for r in OpportunityQuestion.objects.filter(opportunity_id=opportunity_id).order_by("order"):
            data.append({
                "id": str(r.id),
                "question_text": r.question_text,
                "question_type": r.question_type,
                "is_required": r.is_required,
                "order": r.order,
                "config": r.config,
            })
        return data
    rows = await sync_to_async(q)()
    return ok({"questions": rows}, message=f"{len(rows)} questions")


@server.tool()
async def upsert_screening_question_simple(opportunity_id: str, question_text: str, question_type: str, is_required: bool = False, order: int | None = None, config: dict | None = None, id: str | None = None) -> str:
    # Convenience wrapper that builds the payload and reuses the primary upsert
    payload = {
        "id": id,
        "question_text": question_text,
        "question_type": question_type,
        "is_required": is_required,
        "order": order,
        "config": config or {},
    }
    return await upsert_screening_question(opportunity_id=opportunity_id, **payload)


@server.tool()
async def submit_application_answers(application_id: str, answers: list) -> str:
    objs = await sync_to_async(app_service.submit_answers)(application_id, answers)
    return ok({"application_id": application_id, "count": len(objs), "answers": [{"id": str(o.id), "question_id": str(o.question_id), "is_disqualifying": o.is_disqualifying} for o in objs]}, message="Answers submitted")


@server.tool()
async def list_applications_for_profile(profile_id: str, status: str | None = None, limit: int = 50) -> str:
    def q():
        qs = Application.objects.select_related(
            "opportunity__organisation", "current_stage_instance__stage_template"
        ).filter(profile_id=profile_id).order_by("-applied_at")
        if status:
            qs = qs.filter(status=status)
        data = []
        for a in qs[:limit]:
            data.append({
                "id": str(a.id),
                "opportunity": f"{a.opportunity.title} @ {a.opportunity.organisation.name}",
                "status": a.status,
                "current_stage": a.current_stage_instance.stage_template.slug if a.current_stage_instance else None,
                "applied_at": a.applied_at.isoformat(),
            })
        return data
    rows = await sync_to_async(q)()
    return ok({"applications": rows}, message=f"{len(rows)} applications")


@server.tool()
async def bulk_change_stage(application_ids: list, stage_slug: str) -> str:
    successes: list[dict] = []
    failures: list[dict] = []
    for app_id in application_ids:
        try:
            app = await sync_to_async(app_service.change_stage)(app_id, stage_slug)
            successes.append({"application_id": app_id, "status": app.status, "current_stage": app.current_stage_instance.stage_template.slug if app.current_stage_instance else None})
        except Exception as e:
            failures.append({"application_id": app_id, "error": str(e)})
    return ok({"successes": successes, "failures": failures}, message=f"{len(successes)} succeeded, {len(failures)} failed")


@server.tool()
async def schedule_interview_minimal(application_id: str, round_name: str, scheduled_start: str, scheduled_end: str, location_type: str = "virtual", location_details: str = "") -> str:
    from datetime import datetime
    def parse_iso(s: str) -> datetime:
        return datetime.fromisoformat(s.replace("Z", "+00:00")) if isinstance(s, str) else s
    start_dt = parse_iso(scheduled_start)
    end_dt = parse_iso(scheduled_end)
    interview = await sync_to_async(app_service.schedule_interview_minimal)(application_id, round_name, start_dt, end_dt, location_type, location_details)
    return ok({"interview_id": str(interview.id), "application_id": str(interview.application_id), "round_name": interview.round_name, "scheduled_start": interview.scheduled_start.isoformat(), "scheduled_end": interview.scheduled_end.isoformat(), "location_type": interview.location_type}, message="Interview scheduled")


@server.tool()
async def find_candidates_for_opportunity(opportunity_id: str, llm_similarity_threshold: float = 0.7, limit: int | None = 10) -> str:
    result = await eval_service.find_candidates_for_opportunity_async(opportunity_id=opportunity_id, llm_similarity_threshold=llm_similarity_threshold, limit=limit)
    return ok(result, message="Candidates evaluated")


@server.tool()
async def evaluate_candidate_profile(profile_id: str, opportunity_id: str) -> str:
    result = await eval_service.evaluate_single_candidate_async(profile_id=profile_id, opportunity_id=opportunity_id)
    return ok(result, message="Candidate evaluated")


@server.tool()
async def find_opportunities_for_profile(profile_id: str, llm_similarity_threshold: float = 0.7, limit: int | None = 10) -> str:
    result = await eval_service.find_opportunities_for_profile_async(profile_id=profile_id, llm_similarity_threshold=llm_similarity_threshold, limit=limit)
    return ok(result, message="Opportunities evaluated")


@server.tool()
async def analyze_opportunity_fit(profile_id: str, opportunity_id: str) -> str:
    result = await eval_service.analyze_opportunity_fit_async(profile_id=profile_id, opportunity_id=opportunity_id)
    return ok(result, message="Opportunity fit analyzed")


@server.tool()
async def analyze_talent_pool(skill_names: list | None = None) -> str:
    result = await eval_service.analyze_talent_pool_async(skill_names=skill_names)
    return ok(result, message="Talent pool analyzed")


@server.tool()
async def get_learning_recommendations(profile_id: str, target_opportunities: list | None = None) -> str:
    result = await eval_service.get_learning_recommendations_async(profile_id=profile_id, target_opportunities=target_opportunities)
    return ok(result, message="Learning recommendations")


@server.tool()
async def create_profile(first_name: str, last_name: str, email: str, skills: list | None = None, experiences: list | None = None) -> str:
    from apps.profiles.models import Profile, ProfileSkill, ProfileExperience
    from apps.organisations.models import Organisation
    from apps.skills.models import Skill
    from datetime import datetime, date
    try:
        profile = await sync_to_async(Profile.objects.create)(first_name=first_name, last_name=last_name, email=email)
        created_skills = []
        if skills:
            for s in skills:
                skill, _ = await sync_to_async(Skill.objects.get_or_create)(name=s)
                ps = await sync_to_async(ProfileSkill.objects.create)(profile=profile, skill=skill, evidence_level='stated')
                await sync_to_async(ps.ensure_embedding)()
                created_skills.append(s)
        created_exps = []
        if experiences:
            for exp in experiences:
                org, _ = await sync_to_async(Organisation.objects.get_or_create)(name=exp.get('company', 'Unknown Company'))
                sd = exp.get('start_date'); ed = exp.get('end_date')
                if isinstance(sd, str):
                    try:
                        sd = datetime.strptime(sd, '%Y-%m-%d').date()
                    except Exception:
                        sd = date(2020,1,1)
                elif not isinstance(sd, date):
                    sd = date(2020,1,1)
                if isinstance(ed, str):
                    try:
                        ed = datetime.strptime(ed, '%Y-%m-%d').date()
                    except Exception:
                        ed = None
                elif ed and not isinstance(ed, date):
                    ed = None
                await sync_to_async(ProfileExperience.objects.create)(profile=profile, organisation=org, title=exp.get('title','Position'), description=exp.get('description',''), start_date=sd, end_date=ed)
                created_exps.append(f"{exp.get('title','Position')} at {org.name}")
        return ok({"profile_id": str(profile.id), "created_skills": created_skills, "created_experiences": created_exps}, message="Profile created")
    except Exception as e:
        return err("create_failed", str(e))


@server.tool()
async def update_profile(profile_id: str, first_name: str | None = None, last_name: str | None = None, email: str | None = None, add_skills: list | None = None, remove_skills: list | None = None, add_experiences: list | None = None) -> str:
    from apps.profiles.models import Profile, ProfileSkill, ProfileExperience
    from apps.organisations.models import Organisation
    from apps.skills.models import Skill
    from datetime import datetime, date
    try:
        profile = await sync_to_async(Profile.objects.get)(id=profile_id)
    except Profile.DoesNotExist:
        return err("not_found", f"Profile {profile_id} not found")
    updated_fields = []
    try:
        if first_name:
            profile.first_name = first_name; updated_fields.append('first_name')
        if last_name:
            profile.last_name = last_name; updated_fields.append('last_name')
        if email:
            profile.email = email; updated_fields.append('email')
        if updated_fields:
            await sync_to_async(profile.save)()
        added_skills = []
        if add_skills:
            for s in add_skills:
                skill, _ = await sync_to_async(Skill.objects.get_or_create)(name=s)
                exists = await sync_to_async(lambda: ProfileSkill.objects.filter(profile=profile, skill=skill).exists())()
                if not exists:
                    ps = await sync_to_async(ProfileSkill.objects.create)(profile=profile, skill=skill, evidence_level='stated')
                    await sync_to_async(ps.ensure_embedding)()
                    added_skills.append(s)
        removed_skills = []
        if remove_skills:
            for s in remove_skills:
                skill = await sync_to_async(lambda: Skill.objects.filter(name=s).first())()
                if skill:
                    deleted, _ = await sync_to_async(lambda: ProfileSkill.objects.filter(profile=profile, skill=skill).delete())()
                    if deleted:
                        removed_skills.append(s)
        added_exps = []
        if add_experiences:
            for exp in add_experiences:
                org, _ = await sync_to_async(Organisation.objects.get_or_create)(name=exp.get('company','Unknown Company'))
                sd = exp.get('start_date'); ed = exp.get('end_date')
                if isinstance(sd, str):
                    try:
                        sd = datetime.strptime(sd, '%Y-%m-%d').date()
                    except Exception:
                        sd = date(2020,1,1)
                elif not isinstance(sd, date):
                    sd = date(2020,1,1)
                if isinstance(ed, str):
                    try:
                        ed = datetime.strptime(ed, '%Y-%m-%d').date()
                    except Exception:
                        ed = None
                elif ed and not isinstance(ed, date):
                    ed = None
                await sync_to_async(ProfileExperience.objects.create)(profile=profile, organisation=org, title=exp.get('title','Position'), description=exp.get('description',''), start_date=sd, end_date=ed)
                added_exps.append(f"{exp.get('title','Position')} at {org.name}")
        return ok({"profile_id": profile_id, "updated_fields": updated_fields, "added_skills": added_skills, "removed_skills": removed_skills, "added_experiences": added_exps}, message="Profile updated")
    except Exception as e:
        return err("update_failed", str(e))


@server.tool()
async def create_opportunity(title: str, company_name: str, location: str | None = None, description: str | None = None, required_skills: list | None = None, salary_range: str | None = None) -> str:
    from apps.opportunities.models import Opportunity, OpportunitySkill
    from apps.organisations.models import Organisation
    from apps.skills.models import Skill
    try:
        org, _ = await sync_to_async(Organisation.objects.get_or_create)(name=company_name)
        opp = await sync_to_async(Opportunity.objects.create)(title=title, organisation=org, description=description or "")
        created_skills = []
        if required_skills:
            for s in required_skills:
                skill, _ = await sync_to_async(Skill.objects.get_or_create)(name=s)
                osr = await sync_to_async(OpportunitySkill.objects.create)(opportunity=opp, skill=skill, requirement_type='required')
                await sync_to_async(osr.ensure_embedding)()
                created_skills.append(s)
        return ok({"opportunity_id": str(opp.id), "organisation": org.name, "created_required_skills": created_skills}, message="Opportunity created")
    except Exception as e:
        return err("create_failed", str(e))


@server.tool()
async def update_opportunity(opportunity_id: str, title: str | None = None, description: str | None = None, organization_name: str | None = None, add_skills: list | None = None, remove_skills: list | None = None) -> str:
    from apps.opportunities.models import Opportunity, OpportunitySkill
    from apps.organisations.models import Organisation
    from apps.skills.models import Skill
    try:
        opp = await sync_to_async(Opportunity.objects.select_related('organisation').get)(id=opportunity_id)
    except Opportunity.DoesNotExist:
        return err("not_found", f"Opportunity {opportunity_id} not found")
    updated_fields = []
    try:
        if title:
            opp.title = title; updated_fields.append('title')
        if description:
            opp.description = description; updated_fields.append('description')
        if organization_name:
            org, _ = await sync_to_async(Organisation.objects.get_or_create)(name=organization_name)
            opp.organisation = org; updated_fields.append('organisation')
        if updated_fields:
            await sync_to_async(opp.save)()
        added_skills = []
        if add_skills:
            for s in add_skills:
                skill, _ = await sync_to_async(Skill.objects.get_or_create)(name=s)
                exists = await sync_to_async(lambda: OpportunitySkill.objects.filter(opportunity=opp, skill=skill).exists())()
                if not exists:
                    osr = await sync_to_async(OpportunitySkill.objects.create)(opportunity=opp, skill=skill, requirement_type='required')
                    await sync_to_async(osr.ensure_embedding)()
                    added_skills.append(s)
        removed_skills = []
        if remove_skills:
            for s in remove_skills:
                skill = await sync_to_async(lambda: Skill.objects.filter(name=s).first())()
                if skill:
                    deleted, _ = await sync_to_async(lambda: OpportunitySkill.objects.filter(opportunity=opp, skill=skill).delete())()
                    if deleted:
                        removed_skills.append(s)
        return ok({"opportunity_id": opportunity_id, "updated_fields": updated_fields, "added_skills": added_skills, "removed_skills": removed_skills}, message="Opportunity updated")
    except Exception as e:
        return err("update_failed", str(e))


@server.tool()
async def list_langgraph_tools(role: str = "general") -> str:
    role = (role or "general").lower()
    if role == "employer":
        tools = agent_tools.EMPLOYER_TOOLS
    elif role == "candidate":
        tools = agent_tools.CANDIDATE_TOOLS
    else:
        tools = agent_tools.ALL_AGENT_TOOLS
    def tool_info(t):
        schema = None
        if getattr(t, "args_schema", None):
            try:
                schema = t.args_schema.model_json_schema()
            except Exception:
                schema = None
        return {"name": t.name, "description": getattr(t, "description", ""), "schema": schema}
    data = [tool_info(t) for t in tools]
    return ok({"role": role, "tools": data}, message=f"{len(data)} tools")


@server.tool()
async def call_langgraph_tool(tool_name: str, args: dict, role: str = "general") -> str:
    role = (role or "general").lower()
    if role == "employer":
        tools = {t.name: t for t in agent_tools.EMPLOYER_TOOLS}
    elif role == "candidate":
        tools = {t.name: t for t in agent_tools.CANDIDATE_TOOLS}
    else:
        tools = {t.name: t for t in agent_tools.ALL_AGENT_TOOLS}
    tool = tools.get(tool_name)
    if not tool:
        return err("not_found", f"Tool '{tool_name}' not found for role '{role}'", suggestions={"use": "list_langgraph_tools"})
    try:
        # Prefer the standardized Runnable interface if available
        if hasattr(tool, "ainvoke"):
            result = await tool.ainvoke(args or {})
        elif hasattr(tool, "_arun"):
            result = await tool._arun(**(args or {}))  # type: ignore
        else:
            result = tool._run(**(args or {}))  # type: ignore
        return ok({"tool": tool_name, "result": result}, message="tool executed")
    except Exception as e:
        schema = None
        if getattr(tool, "args_schema", None):
            try:
                schema = tool.args_schema.model_json_schema()
            except Exception:
                schema = None
        return err(
            "tool_failed",
            f"{tool_name} error: {e}",
            suggestions={"schema": schema, "tip": "Match property names and required fields"}
        )


@server.tool()
async def agent_invoke(message: str, role: str = "general", session_id: str | None = None) -> str:
    try:
        if role.lower() == "employer":
            agent = create_employer_agent()
        elif role.lower() == "candidate":
            agent = create_candidate_agent()
        else:
            agent = create_general_agent()
        res = agent.invoke(message=message, session_id=session_id or "default")
        return ok({"response": res}, message="agent responded")
    except Exception as e:
        return err("agent_failed", str(e))


if __name__ == "__main__":
    server.run()
