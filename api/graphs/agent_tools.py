"""
Granular async tools for TalentCo agents.
Implements specific, focused tools for employer and candidate agent roles.
"""

import json
from typing import Optional, List, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from pydantic import BaseModel, Field

# Django setup for tools
import os
import django
from django.conf import settings
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'talentco.settings')
if not settings.configured:
    django.setup()

from apps.evaluations.services import EvaluationService
from apps.profiles.models import Profile
from apps.opportunities.models import Opportunity

# Helper JSON envelopes for tool outputs
import json as _json_helpers

def _ok(data: dict, message: str = "") -> str:
    return _json_helpers.dumps({"status": "ok", "data": data, "message": message})

def _err(code: str, message: str, *, hints: list | None = None, fields: list | None = None, suggestions: dict | None = None) -> str:
    return _json_helpers.dumps({
        "status": "error",
        "code": code,
        "message": message,
        "hints": hints or [],
        "fields": fields or [],
        "suggestions": suggestions or {},
    })

# ================================================================
# EMPLOYER AGENT TOOLS - For hiring managers and recruiters
# ================================================================

class FindCandidatesInput(BaseModel):
    """Input for finding candidates for an opportunity."""
    opportunity_id: str = Field(description="UUID of the opportunity to find candidates for")
    llm_similarity_threshold: float = Field(
        default=0.7,
        description="Similarity threshold for LLM judging (0.0-1.0, default 0.7)"
    )
    limit: Optional[int] = Field(
        default=10,
        description="Maximum number of candidates to return (default 10)"
    )


class FindCandidatesForOpportunityTool(BaseTool):
    """Find the best candidates for a job opportunity."""
    
    name: str = "find_candidates_for_opportunity"
    description: str = """Find and rank the best candidates for a specific job opportunity.
    
    Uses TalentCo's multi-stage evaluation pipeline:
    1. Structured matching (skills overlap)
    2. Semantic similarity (vector embeddings)  
    3. LLM judge evaluation for top matches
    
    Returns ranked candidates with detailed scores and reasoning."""
    
    args_schema: type[BaseModel] = FindCandidatesInput
    
    async def _arun(
        self,
        opportunity_id: str,
        llm_similarity_threshold: float = 0.7,
        limit: Optional[int] = 10,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the async find candidates tool."""
        try:
            service = EvaluationService()
            result = await service.find_candidates_for_opportunity_async(
                opportunity_id=opportunity_id,
                llm_similarity_threshold=llm_similarity_threshold,
                limit=limit
            )
            
            # Format for agent consumption
            summary = f"""Found {result['total_candidates_evaluated']} candidates for opportunity.
Top {len(result['top_matches'])} matches (LLM judged: {result['llm_judged_count']}):

"""
            
            for match in result['top_matches'][:5]:  # Show top 5 in summary
                summary += f"#{match['rank']}: {match['candidate_name']} (Score: {match['final_score']:.3f})\n"
                if match['was_llm_judged'] and match['llm_reasoning']:
                    summary += f"   💭 {match['llm_reasoning'][:100]}...\n"
                summary += f"   📊 Structured: {match['structured_score']:.3f} | Semantic: {match['semantic_score']:.3f}\n\n"
            
            return summary + f"\nFull results available via evaluation_set_id: {result['evaluation_set_id']}"
            
        except Exception as e:
            return f"Error finding candidates: {str(e)}"
    
    def _run(self, *args, **kwargs) -> str:
        """Sync version not implemented - use async only."""
        return "This tool requires async execution. Use _arun method."


class EvaluateCandidateInput(BaseModel):
    """Input for detailed candidate evaluation."""
    profile_id: str = Field(description="UUID of the candidate profile to evaluate")
    opportunity_id: str = Field(description="UUID of the opportunity to evaluate against")


class EvaluateCandidateProfileTool(BaseTool):
    """Perform detailed evaluation of a specific candidate for an opportunity."""
    
    name: str = "evaluate_candidate_profile"
    description: str = """Perform deep analysis of a specific candidate for an opportunity.
    
    Provides detailed breakdown of:
    - Structured matching scores
    - Semantic similarity analysis
    - LLM judge evaluation with reasoning
    - Skill fit analysis
    - Experience relevance assessment
    
    Use this for in-depth candidate assessment."""
    
    args_schema: type[BaseModel] = EvaluateCandidateInput
    
    async def _arun(
        self,
        profile_id: str,
        opportunity_id: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the async candidate evaluation tool."""
        try:
            service = EvaluationService()
            result = await service.evaluate_single_candidate_async(
                profile_id=profile_id,
                opportunity_id=opportunity_id
            )
            
            scores = result['detailed_scores']
            
            evaluation = f"""🎯 Detailed Candidate Evaluation: {result['candidate_name']}

📊 Matching Scores:
• Combined Score: {scores['combined_score']:.3f} ({self._score_interpretation(scores['combined_score'])})
• Structured Match: {scores['structured_match']:.3f}
• Semantic Similarity: {scores['semantic_similarity']:.3f} 
• LLM Judge Score: {scores['llm_judge_score']:.3f}

💭 LLM Assessment:
{scores['llm_reasoning']}

🔍 Skill Analysis:
{result['skill_analysis']['skill_overlap_percentage']:.1f}% skill overlap

📈 Experience Analysis:  
{result['experience_analysis']['experience_relevance']:.1f}% experience relevance
"""
            
            return evaluation
            
        except Exception as e:
            return f"Error evaluating candidate: {str(e)}"
    
    def _run(self, *args, **kwargs) -> str:
        """Sync version not implemented - use async only."""
        return "This tool requires async execution. Use _arun method."
    
    def _score_interpretation(self, score: float) -> str:
        """Interpret score for human readability."""
        if score >= 0.9: return "Excellent Match"
        elif score >= 0.8: return "Strong Match"
        elif score >= 0.7: return "Good Match"
        elif score >= 0.6: return "Moderate Match"
        else: return "Weak Match"


class AnalyzeTalentPoolInput(BaseModel):
    """Input for talent pool analysis."""
    skill_names: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific skills to analyze (if None, analyzes all skills)"
    )


class CreateOpportunityInput(BaseModel):
    """Input for creating a new job opportunity."""
    title: str = Field(description="Job title/role name")
    company_name: str = Field(description="Company/organization name")
    location: Optional[str] = Field(
        default="Not specified",
        description="Job location (e.g., 'Remote', 'New York, NY')"
    )
    description: Optional[str] = Field(
        default="Job description not provided",
        description="Detailed job description"
    )
    required_skills: Optional[List[str]] = Field(
        default=None,
        description="List of required skills for the role"
    )
    salary_range: Optional[str] = Field(
        default=None,
        description="Salary range (e.g., '$80,000 - $120,000')"
    )


class CreateOpportunityTool(BaseTool):
    """Create a new job opportunity in the system."""
    
    name: str = "create_opportunity"
    description: str = """Create a new job opportunity that can be used for candidate matching.
    
    This tool:
    - Creates the opportunity in the database
    - Sets up required skills if provided
    - Returns the opportunity ID for use with other tools
    - Enables candidate matching via find_candidates_for_opportunity
    
    Use this when you need to post a new job or when an opportunity mentioned 
    in conversation doesn't exist in the system yet."""
    
    args_schema: type[BaseModel] = CreateOpportunityInput
    
    async def _arun(
        self,
        title: str,
        company_name: str,
        location: Optional[str] = "Not specified",
        description: Optional[str] = "Job description not provided",
        required_skills: Optional[List[str]] = None,
        salary_range: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the async create opportunity tool."""
        try:
            from asgiref.sync import sync_to_async
            from apps.opportunities.models import Opportunity, OpportunitySkill
            from apps.organisations.models import Organisation
            from apps.skills.models import Skill
            
            # Get or create the organization
            get_or_create_org = sync_to_async(Organisation.objects.get_or_create)
            organisation, created = await get_or_create_org(
                name=company_name
            )
            
            # Create the opportunity  
            create_opportunity = sync_to_async(Opportunity.objects.create)
            opportunity = await create_opportunity(
                title=title,
                organisation=organisation,
                description=description
            )
            
            # Add required skills if provided
            if required_skills:
                created_skills = []
                for skill_name in required_skills:
                    # Get or create the skill (Skill model only has name field)
                    get_or_create_skill = sync_to_async(Skill.objects.get_or_create)
                    skill, skill_created = await get_or_create_skill(
                        name=skill_name
                    )
                    
                    # Create opportunity skill relationship
                    create_opp_skill = sync_to_async(OpportunitySkill.objects.create)
                    opp_skill = await create_opp_skill(
                        opportunity=opportunity,
                        skill=skill,
                        requirement_type='required'
                    )
                    
                    # Generate embedding for new opportunity skill
                    ensure_embedding = sync_to_async(opp_skill.ensure_embedding)
                    await ensure_embedding()
                    
                    created_skills.append(skill_name)
                
                skills_summary = f"\n• Required skills: {', '.join(created_skills)}"
            else:
                skills_summary = ""
            
            result = f"""✅ Job opportunity created successfully!

📋 Opportunity Details:
• Title: {title}
• Company: {company_name}
• Opportunity ID: {opportunity.id}{skills_summary}

🎯 Next Steps:
You can now use find_candidates_for_opportunity with ID: {opportunity.id}
"""
            
            return result
            
        except Exception as e:
            return f"Error creating opportunity: {str(e)}"
    
    def _run(self, *args, **kwargs) -> str:
        """Sync version not implemented - use async only."""
        return "This tool requires async execution. Use _arun method."


class AnalyzeTalentPoolTool(BaseTool):
    """Analyze the available talent pool and skill market insights."""
    
    name: str = "analyze_talent_pool"
    description: str = """Analyze the current talent pool to understand:
    - Total candidate availability
    - Experience level distribution
    - Skill availability and scarcity
    - Market insights for hiring strategy
    
    Helpful for understanding talent market dynamics."""
    
    args_schema: type[BaseModel] = AnalyzeTalentPoolInput
    
    async def _arun(
        self,
        skill_names: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the async talent pool analysis tool."""
        try:
            service = EvaluationService()
            result = await service.analyze_talent_pool_async(skill_names=skill_names)
            
            exp_dist = result['experience_distribution']
            insights = result['market_insights']
            
            analysis = f"""📊 Talent Pool Analysis ({result['total_candidates']} candidates)

👥 Experience Distribution:
• Junior (0-3 years): {exp_dist['junior']} candidates ({exp_dist['junior']/result['total_candidates']*100:.1f}%)
• Mid-level (3-7 years): {exp_dist['mid']} candidates ({exp_dist['mid']/result['total_candidates']*100:.1f}%)
• Senior (7-15 years): {exp_dist['senior']} candidates ({exp_dist['senior']/result['total_candidates']*100:.1f}%)
• Executive (15+ years): {exp_dist['executive']} candidates ({exp_dist['executive']/result['total_candidates']*100:.1f}%)

🔥 Most In-Demand Skills:
"""
            for skill, data in insights['most_common_skills'][:5]:
                analysis += f"• {skill}: {data['count']} candidates\n"
            
            analysis += f"\n💎 Scarce Skills (high value):\n"
            for skill in insights['skill_scarcity'][:5]:
                analysis += f"• {skill}\n"
            
            return analysis
            
        except Exception as e:
            return f"Error analyzing talent pool: {str(e)}"
    
    def _run(self, *args, **kwargs) -> str:
        """Sync version not implemented - use async only."""
        return "This tool requires async execution. Use _arun method."


# ================================================================
# CANDIDATE AGENT TOOLS - For job seekers and career development
# ================================================================

class FindOpportunitiesInput(BaseModel):
    """Input for finding opportunities for a profile."""
    profile_id: str = Field(description="UUID of the profile to find opportunities for")
    llm_similarity_threshold: float = Field(
        default=0.7,
        description="Similarity threshold for LLM judging (0.0-1.0, default 0.7)"
    )
    limit: Optional[int] = Field(
        default=10,
        description="Maximum number of opportunities to return (default 10)"
    )


class FindOpportunitiesForProfileTool(BaseTool):
    """Find the best job opportunities for a candidate profile."""
    
    name: str = "find_opportunities_for_profile"
    description: str = """Find and rank the best job opportunities for a specific candidate.
    
    Uses TalentCo's multi-stage evaluation pipeline:
    1. Structured matching (skills overlap)
    2. Semantic similarity (vector embeddings)
    3. LLM judge evaluation for top matches
    
    Returns ranked opportunities with detailed scores and reasoning."""
    
    args_schema: type[BaseModel] = FindOpportunitiesInput
    
    async def _arun(
        self,
        profile_id: str,
        llm_similarity_threshold: float = 0.7,
        limit: Optional[int] = 10,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the async find opportunities tool."""
        try:
            service = EvaluationService()
            result = await service.find_opportunities_for_profile_async(
                profile_id=profile_id,
                llm_similarity_threshold=llm_similarity_threshold,
                limit=limit
            )
            
            # Format for agent consumption
            summary = f"""Found {result['total_opportunities_evaluated']} opportunities for your profile.
Top {len(result['top_matches'])} matches (LLM analyzed: {result['llm_judged_count']}):

"""
            
            for match in result['top_matches'][:5]:  # Show top 5 in summary
                summary += f"#{match['rank']}: {match['role_title']} at {match['company_name']} (Score: {match['final_score']:.3f})\n"
                if match['was_llm_judged'] and match['llm_reasoning']:
                    summary += f"   💭 {match['llm_reasoning'][:100]}...\n"
                summary += f"   📊 Structured: {match['structured_score']:.3f} | Semantic: {match['semantic_score']:.3f}\n\n"
            
            return summary + f"\nFull results available via evaluation_set_id: {result['evaluation_set_id']}"
            
        except Exception as e:
            return f"Error finding opportunities: {str(e)}"
    
    def _run(self, *args, **kwargs) -> str:
        """Sync version not implemented - use async only."""
        return "This tool requires async execution. Use _arun method."


class AnalyzeOpportunityFitInput(BaseModel):
    """Input for analyzing opportunity fit."""
    profile_id: str = Field(description="UUID of the candidate profile")
    opportunity_id: str = Field(description="UUID of the opportunity to analyze")


class AnalyzeOpportunityFitTool(BaseTool):
    """Analyze how well a specific opportunity fits a candidate's profile."""
    
    name: str = "analyze_opportunity_fit"
    description: str = """Perform detailed analysis of opportunity fit for a candidate.
    
    Provides detailed breakdown of:
    - Overall fit assessment 
    - Skill gap analysis
    - Experience relevance
    - LLM assessment with reasoning
    
    Use this to help candidates understand opportunity compatibility."""
    
    args_schema: type[BaseModel] = AnalyzeOpportunityFitInput
    
    async def _arun(
        self,
        profile_id: str,
        opportunity_id: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the async opportunity fit analysis tool."""
        try:
            service = EvaluationService()
            result = await service.analyze_opportunity_fit_async(
                profile_id=profile_id,
                opportunity_id=opportunity_id
            )
            
            fit = result['fit_analysis']
            
            analysis = f"""🎯 Opportunity Fit Analysis: {result['role_title']} at {result['company_name']}

📊 Fit Assessment:
• Overall Fit: {fit['combined_score']:.3f} ({self._score_interpretation(fit['combined_score'])})
• Structured Match: {fit['structured_match']:.3f}
• Semantic Similarity: {fit['semantic_similarity']:.3f}
• LLM Assessment: {fit['llm_assessment_score']:.3f}

💭 AI Analysis:
{fit['llm_reasoning']}

🔍 Skill Gap Analysis:
{result['skill_gap_analysis']['skill_overlap_percentage']:.1f}% of your skills match this role

📈 Experience Relevance:
{result['experience_relevance']['experience_relevance']:.1f}% of your experience is relevant
"""
            
            return analysis
            
        except Exception as e:
            return f"Error analyzing opportunity fit: {str(e)}"
    
    def _run(self, *args, **kwargs) -> str:
        """Sync version not implemented - use async only."""
        return "This tool requires async execution. Use _arun method."
    
    def _score_interpretation(self, score: float) -> str:
        """Interpret score for human readability."""
        if score >= 0.9: return "Excellent Fit"
        elif score >= 0.8: return "Strong Fit"
        elif score >= 0.7: return "Good Fit"
        elif score >= 0.6: return "Moderate Fit"
        else: return "Weak Fit"


class CreateProfileInput(BaseModel):
    """Input for creating a new candidate profile."""
    first_name: str = Field(description="Candidate's first name")
    last_name: str = Field(description="Candidate's last name")
    email: str = Field(description="Candidate's email address")
    skills: Optional[List[str]] = Field(
        default=None,
        description="List of candidate's skills"
    )
    experiences: Optional[List[Dict]] = Field(
        default=None,
        description="List of work experiences with title, company, description, start_date, end_date"
    )


class CreateProfileTool(BaseTool):
    """Create a new candidate profile in the system."""
    
    name: str = "create_profile"
    description: str = """Create a new candidate profile that can be used for opportunity matching.
    
    This tool:
    - Creates the profile in the database
    - Sets up skills and experiences if provided
    - Returns the profile ID for use with other tools
    - Enables opportunity matching via find_opportunities_for_profile
    
    Use this when you need to register a new candidate or when candidate information 
    mentioned in conversation doesn't exist in the system yet."""
    
    args_schema: type[BaseModel] = CreateProfileInput
    
    async def _arun(
        self,
        first_name: str,
        last_name: str,
        email: str,
        skills: Optional[List[str]] = None,
        experiences: Optional[List[Dict]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the async create profile tool."""
        try:
            from asgiref.sync import sync_to_async
            from apps.profiles.models import Profile, ProfileSkill, ProfileExperience
            from apps.organisations.models import Organisation
            from apps.skills.models import Skill
            from datetime import datetime, date
            
            # Create the profile
            create_profile = sync_to_async(Profile.objects.create)
            profile = await create_profile(
                first_name=first_name,
                last_name=last_name,
                email=email
            )
            
            created_skills = []
            created_experiences = []
            
            # Add skills if provided
            if skills:
                for skill_name in skills:
                    # Get or create the skill
                    get_or_create_skill = sync_to_async(Skill.objects.get_or_create)
                    skill, skill_created = await get_or_create_skill(name=skill_name)
                    
                    # Create profile skill relationship
                    create_profile_skill = sync_to_async(ProfileSkill.objects.create)
                    profile_skill = await create_profile_skill(
                        profile=profile,
                        skill=skill,
                        evidence_level='stated'  # Default evidence level
                    )
                    
                    # Generate embedding for new profile skill
                    ensure_embedding = sync_to_async(profile_skill.ensure_embedding)
                    await ensure_embedding()
                    
                    created_skills.append(skill_name)
            
            # Add experiences if provided
            if experiences:
                for exp_data in experiences:
                    # Get or create the organization
                    company_name = exp_data.get('company', 'Unknown Company')
                    get_or_create_org = sync_to_async(Organisation.objects.get_or_create)
                    organisation, created = await get_or_create_org(name=company_name)
                    
                    # Parse dates
                    start_date = exp_data.get('start_date')
                    end_date = exp_data.get('end_date')
                    
                    if isinstance(start_date, str):
                        try:
                            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
                        except:
                            start_date = date(2020, 1, 1)  # Default fallback
                    elif not isinstance(start_date, date):
                        start_date = date(2020, 1, 1)
                    
                    if end_date and isinstance(end_date, str):
                        try:
                            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
                        except:
                            end_date = None
                    elif end_date and not isinstance(end_date, date):
                        end_date = None
                    
                    # Create profile experience
                    create_profile_exp = sync_to_async(ProfileExperience.objects.create)
                    await create_profile_exp(
                        profile=profile,
                        organisation=organisation,
                        title=exp_data.get('title', 'Position'),
                        description=exp_data.get('description', ''),
                        start_date=start_date,
                        end_date=end_date
                    )
                    created_experiences.append(f"{exp_data.get('title', 'Position')} at {company_name}")
            
            # Build response
            skills_summary = f"\n• Skills: {', '.join(created_skills)}" if created_skills else ""
            exp_summary = f"\n• Experiences: {', '.join(created_experiences)}" if created_experiences else ""
            
            result = f"""✅ Candidate profile created successfully!

👤 Profile Details:
• Name: {first_name} {last_name}
• Email: {email}
• Profile ID: {profile.id}{skills_summary}{exp_summary}

🎯 Next Steps:
You can now use find_opportunities_for_profile with ID: {profile.id}
"""
            
            return result
            
        except Exception as e:
            return f"Error creating profile: {str(e)}"
    
    def _run(self, *args, **kwargs) -> str:
        """Sync version not implemented - use async only."""
        return "This tool requires async execution. Use _arun method."


class UpdateProfileInput(BaseModel):
    """Input for updating an existing candidate profile."""
    profile_id: str = Field(description="UUID of the profile to update")
    first_name: Optional[str] = Field(default=None, description="New first name (optional)")
    last_name: Optional[str] = Field(default=None, description="New last name (optional)")
    email: Optional[str] = Field(default=None, description="New email address (optional)")
    add_skills: Optional[List[str]] = Field(
        default=None,
        description="List of skills to add to the profile"
    )
    remove_skills: Optional[List[str]] = Field(
        default=None,
        description="List of skills to remove from the profile"
    )
    add_experiences: Optional[List[Dict]] = Field(
        default=None,
        description="List of work experiences to add with title, company, description, start_date, end_date"
    )


class UpdateProfileTool(BaseTool):
    """Update an existing candidate profile."""
    
    name: str = "update_profile"
    description: str = """Update an existing candidate profile with new information.
    
    This tool can:
    - Update basic profile information (name, email)
    - Add new skills to the profile
    - Remove existing skills from the profile
    - Add new work experiences
    - Generate embeddings for any new data
    
    Use this when you need to modify existing candidate information or add new qualifications."""
    
    args_schema: type[BaseModel] = UpdateProfileInput
    
    async def _arun(
        self,
        profile_id: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        email: Optional[str] = None,
        add_skills: Optional[List[str]] = None,
        remove_skills: Optional[List[str]] = None,
        add_experiences: Optional[List[Dict]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the async update profile tool."""
        try:
            from asgiref.sync import sync_to_async
            from apps.profiles.models import Profile, ProfileSkill, ProfileExperience
            from apps.organisations.models import Organisation
            from apps.skills.models import Skill
            from datetime import datetime, date
            
            # Get the existing profile
            get_profile = sync_to_async(Profile.objects.get)
            try:
                profile = await get_profile(id=profile_id)
            except Profile.DoesNotExist:
                return f"❌ Profile with ID {profile_id} not found."
            
            updated_fields = []
            
            # Update basic profile information
            profile_updated = False
            if first_name:
                profile.first_name = first_name
                profile_updated = True
                updated_fields.append(f"First name: {first_name}")
            
            if last_name:
                profile.last_name = last_name
                profile_updated = True
                updated_fields.append(f"Last name: {last_name}")
            
            if email:
                profile.email = email
                profile_updated = True
                updated_fields.append(f"Email: {email}")
            
            if profile_updated:
                save_profile = sync_to_async(profile.save)
                await save_profile()
            
            # Add new skills
            added_skills = []
            if add_skills:
                for skill_name in add_skills:
                    # Get or create the skill
                    get_or_create_skill = sync_to_async(Skill.objects.get_or_create)
                    skill, skill_created = await get_or_create_skill(name=skill_name)
                    
                    # Check if skill already exists for this profile
                    skill_exists = sync_to_async(
                        lambda: ProfileSkill.objects.filter(profile=profile, skill=skill).exists()
                    )
                    if not await skill_exists():
                        # Create new profile skill relationship
                        create_profile_skill = sync_to_async(ProfileSkill.objects.create)
                        profile_skill = await create_profile_skill(
                            profile=profile,
                            skill=skill,
                            evidence_level='stated'
                        )
                        
                        # Generate embedding for new profile skill
                        ensure_embedding = sync_to_async(profile_skill.ensure_embedding)
                        await ensure_embedding()
                        
                        added_skills.append(skill_name)
            
            # Remove skills
            removed_skills = []
            if remove_skills:
                for skill_name in remove_skills:
                    # Find the skill
                    get_skill = sync_to_async(
                        lambda sn=skill_name: Skill.objects.filter(name=sn).first()
                    )
                    skill = await get_skill()
                    
                    if skill:
                        # Remove the ProfileSkill relationship
                        delete_profile_skill = sync_to_async(
                            lambda: ProfileSkill.objects.filter(profile=profile, skill=skill).delete()
                        )
                        deleted_count, _ = await delete_profile_skill()
                        if deleted_count > 0:
                            removed_skills.append(skill_name)
            
            # Add new experiences
            added_experiences = []
            if add_experiences:
                for exp_data in add_experiences:
                    # Get or create the organization
                    company_name = exp_data.get('company', 'Unknown Company')
                    get_or_create_org = sync_to_async(Organisation.objects.get_or_create)
                    organisation, created = await get_or_create_org(name=company_name)
                    
                    # Parse dates
                    start_date = exp_data.get('start_date')
                    end_date = exp_data.get('end_date')
                    
                    if isinstance(start_date, str):
                        try:
                            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
                        except:
                            start_date = date(2020, 1, 1)  # Default fallback
                    elif not isinstance(start_date, date):
                        start_date = date(2020, 1, 1)
                    
                    if end_date and isinstance(end_date, str):
                        try:
                            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
                        except:
                            end_date = None
                    elif end_date and not isinstance(end_date, date):
                        end_date = None
                    
                    # Create profile experience
                    create_profile_exp = sync_to_async(ProfileExperience.objects.create)
                    await create_profile_exp(
                        profile=profile,
                        organisation=organisation,
                        title=exp_data.get('title', 'Position'),
                        description=exp_data.get('description', ''),
                        start_date=start_date,
                        end_date=end_date
                    )
                    added_experiences.append(f"{exp_data.get('title', 'Position')} at {company_name}")
            
            # Build response
            result_parts = [f"✅ Profile updated successfully for {profile.first_name} {profile.last_name}"]
            
            if updated_fields:
                result_parts.append(f"\n📝 Updated: {', '.join(updated_fields)}")
            
            if added_skills:
                result_parts.append(f"\n➕ Added skills: {', '.join(added_skills)}")
            
            if removed_skills:
                result_parts.append(f"\n➖ Removed skills: {', '.join(removed_skills)}")
            
            if added_experiences:
                result_parts.append(f"\n💼 Added experiences: {', '.join(added_experiences)}")
            
            if not (updated_fields or added_skills or removed_skills or added_experiences):
                result_parts.append("\n📋 No changes were made to the profile.")
            
            return ''.join(result_parts)
            
        except Exception as e:
            return f"Error updating profile: {str(e)}"
    
    def _run(self, *args, **kwargs) -> str:
        """Sync version not implemented - use async only."""
        return "This tool requires async execution. Use _arun method."


class GetLearningRecommendationsInput(BaseModel):
    """Input for learning recommendations."""
    profile_id: str = Field(description="UUID of the candidate profile")
    target_opportunities: Optional[List[str]] = Field(
        default=None,
        description="Optional list of opportunity IDs to analyze (if None, analyzes all opportunities)"
    )


class GetLearningRecommendationsTool(BaseTool):
    """Get personalized learning and skill development recommendations."""
    
    name: str = "get_learning_recommendations"
    description: str = """Analyze skill gaps and provide personalized learning recommendations.
    
    Provides:
    - Current skill inventory
    - Priority skill recommendations based on market demand
    - Learning path suggestions (immediate, medium-term, advanced)
    - Impact assessment for each recommended skill
    
    Helps candidates plan their skill development strategy."""
    
    args_schema: type[BaseModel] = GetLearningRecommendationsInput
    
    async def _arun(
        self,
        profile_id: str,
        target_opportunities: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the async learning recommendations tool."""
        try:
            service = EvaluationService()
            result = await service.get_learning_recommendations_async(
                profile_id=profile_id,
                target_opportunities=target_opportunities
            )
            
            gap_analysis = result['skill_gap_analysis']
            learning_path = result['learning_path']
            
            recommendations = f"""📚 Personalized Learning Recommendations

📊 Current State:
• Your skills: {result['current_skills_count']} 
• Skills gaps identified: {gap_analysis['missing_skills_count']}

🎯 Priority Recommendations:
"""
            
            for rec in gap_analysis['priority_recommendations'][:5]:
                impact_emoji = "🔥" if rec['impact'] == 'high' else "📈"
                recommendations += f"{impact_emoji} {rec['skill_name']} ({rec['impact']} impact)\n"
                recommendations += f"   • Appears in {rec['opportunity_count']} opportunities\n"
                if rec['example_opportunities']:
                    recommendations += f"   • Example roles: {', '.join([opp['title'] for opp in rec['example_opportunities'][:2]])}\n"
                recommendations += "\n"
            
            recommendations += f"""🛤️ Suggested Learning Path:

🚀 Immediate Focus (next 3 months):
{', '.join(learning_path['immediate_focus'])}

📈 Medium Term (3-6 months):
{', '.join(learning_path['medium_term'])}

🎓 Advanced (6+ months):
{', '.join(learning_path['advanced'])}
"""
            
            return recommendations
            
        except Exception as e:
            return f"Error generating learning recommendations: {str(e)}"
    
    def _run(self, *args, **kwargs) -> str:
        """Sync version not implemented - use async only."""
        return "This tool requires async execution. Use _arun method."


class UpdateOpportunityInput(BaseModel):
    """Input for updating an existing job opportunity."""
    opportunity_id: str = Field(description="UUID of the opportunity to update")
    title: Optional[str] = Field(default=None, description="New job title (optional)")
    description: Optional[str] = Field(default=None, description="New job description (optional)")
    organization_name: Optional[str] = Field(default=None, description="New organization name (optional)")
    add_skills: Optional[List[str]] = Field(
        default=None,
        description="List of required skills to add to the opportunity"
    )
    remove_skills: Optional[List[str]] = Field(
        default=None,
        description="List of skills to remove from the opportunity"
    )


class UpdateOpportunityTool(BaseTool):
    """Update an existing job opportunity."""
    
    name: str = "update_opportunity"
    description: str = """Update an existing job opportunity with new information.
    
    This tool can:
    - Update job title and description
    - Change organization information
    - Add new required skills
    - Remove existing skills
    - Generate embeddings for any new data
    
    Use this when you need to modify existing job postings or update requirements."""
    
    args_schema: type[BaseModel] = UpdateOpportunityInput
    
    async def _arun(
        self,
        opportunity_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        organization_name: Optional[str] = None,
        add_skills: Optional[List[str]] = None,
        remove_skills: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the async update opportunity tool."""
        try:
            from asgiref.sync import sync_to_async
            from apps.opportunities.models import Opportunity, OpportunitySkill
            from apps.organisations.models import Organisation
            from apps.skills.models import Skill
            
            # Get the existing opportunity
            get_opportunity = sync_to_async(Opportunity.objects.select_related('organisation').get)
            try:
                opportunity = await get_opportunity(id=opportunity_id)
            except Opportunity.DoesNotExist:
                return f"❌ Opportunity with ID {opportunity_id} not found."
            
            updated_fields = []
            
            # Update basic opportunity information
            opportunity_updated = False
            if title:
                opportunity.title = title
                opportunity_updated = True
                updated_fields.append(f"Title: {title}")
            
            if description:
                opportunity.description = description
                opportunity_updated = True
                updated_fields.append(f"Description updated")
            
            # Update organization if specified
            if organization_name:
                get_or_create_org = sync_to_async(Organisation.objects.get_or_create)
                organisation, created = await get_or_create_org(name=organization_name)
                opportunity.organisation = organisation
                opportunity_updated = True
                updated_fields.append(f"Organization: {organization_name}")
            
            if opportunity_updated:
                save_opportunity = sync_to_async(opportunity.save)
                await save_opportunity()
            
            # Add new skills
            added_skills = []
            if add_skills:
                for skill_name in add_skills:
                    # Get or create the skill
                    get_or_create_skill = sync_to_async(Skill.objects.get_or_create)
                    skill, skill_created = await get_or_create_skill(name=skill_name)
                    
                    # Check if skill already exists for this opportunity
                    skill_exists = sync_to_async(
                        lambda: OpportunitySkill.objects.filter(opportunity=opportunity, skill=skill).exists()
                    )
                    if not await skill_exists():
                        # Create new opportunity skill relationship
                        create_opp_skill = sync_to_async(OpportunitySkill.objects.create)
                        opp_skill = await create_opp_skill(
                            opportunity=opportunity,
                            skill=skill,
                            requirement_type='required'  # Default to required
                        )
                        
                        # Generate embedding for new opportunity skill
                        ensure_embedding = sync_to_async(opp_skill.ensure_embedding)
                        await ensure_embedding()
                        
                        added_skills.append(skill_name)
            
            # Remove skills
            removed_skills = []
            if remove_skills:
                for skill_name in remove_skills:
                    # Find the skill
                    get_skill = sync_to_async(
                        lambda sn=skill_name: Skill.objects.filter(name=sn).first()
                    )
                    skill = await get_skill()
                    
                    if skill:
                        # Remove the OpportunitySkill relationship
                        delete_opp_skill = sync_to_async(
                            lambda: OpportunitySkill.objects.filter(opportunity=opportunity, skill=skill).delete()
                        )
                        deleted_count, _ = await delete_opp_skill()
                        if deleted_count > 0:
                            removed_skills.append(skill_name)
            
            # Build response
            result_parts = [f"✅ Opportunity updated successfully: {opportunity.title} at {opportunity.organisation.name}"]
            
            if updated_fields:
                result_parts.append(f"\n📝 Updated: {', '.join(updated_fields)}")
            
            if added_skills:
                result_parts.append(f"\n➕ Added skills: {', '.join(added_skills)}")
            
            if removed_skills:
                result_parts.append(f"\n➖ Removed skills: {', '.join(removed_skills)}")
            
            if not (updated_fields or added_skills or removed_skills):
                result_parts.append("\n📋 No changes were made to the opportunity.")
            
            return ''.join(result_parts)
            
        except Exception as e:
            return f"Error updating opportunity: {str(e)}"
    
    def _run(self, *args, **kwargs) -> str:
        """Sync version not implemented - use async only."""
        return "This tool requires async execution. Use _arun method."


# ================================================================
# TOOL COLLECTIONS FOR AGENT ROLES
# ================================================================

# Employer agent tools
EMPLOYER_TOOLS = [
    FindCandidatesForOpportunityTool(),
    EvaluateCandidateProfileTool(),
    CreateOpportunityTool(),
    UpdateOpportunityTool(),
    AnalyzeTalentPoolTool()
]

# Candidate agent tools  
CANDIDATE_TOOLS = [
    FindOpportunitiesForProfileTool(),
    AnalyzeOpportunityFitTool(),
    CreateProfileTool(),
    UpdateProfileTool(),
    GetLearningRecommendationsTool()
]

# Combined tools for flexible agent configuration
ALL_AGENT_TOOLS = EMPLOYER_TOOLS + CANDIDATE_TOOLS


# ===============================
# APPLICATIONS (ATS) TOOLS
# ===============================
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

try:
    from apps.applications.services import ApplicationService
except Exception:
    ApplicationService = None  # Avoid import error before migrations


class CreateApplicationInput(BaseModel):
    profile_id: str = Field(description="UUID of the profile applying")
    opportunity_id: str = Field(description="UUID of the target opportunity")
    source: Optional[str] = Field(default="direct", description="Application source")
    answers: Optional[List[Dict[str, Any]]] = Field(default=None, description="Screening answers payload")


class CreateApplicationTool(BaseTool):
    name: str = "create_application_for_profile"
    description: str = (
        "Create an application for a profile against an opportunity; validates screening and snapshots evaluation.\n\n"
        "Examples:\n"
        "{\"profile_id\": \"PROFILE_UUID\", \"opportunity_id\": \"OPPORTUNITY_UUID\"}\n"
        "{\"profile_id\": \"PROFILE_UUID\", \"opportunity_id\": \"OPPORTUNITY_UUID\", \"answers\": ["
        "{\"question_id\": \"Q_UUID\", \"answer_text\": \"10\"}]}"
    )
    args_schema: type[BaseModel] = CreateApplicationInput

    async def _arun(
        self,
        profile_id: str,
        opportunity_id: str,
        source: str = "direct",
        answers: Optional[List[Dict[str, Any]]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if ApplicationService is None:
            return _err("unavailable", "Applications service unavailable. Ensure migrations are applied.")
        service = ApplicationService()
        from asgiref.sync import sync_to_async
        result = await sync_to_async(service.create_application)(profile_id, opportunity_id, source, answers)
        app = result.application
        return _ok({
            "application_id": str(app.id),
            "status": app.status,
            "result": "created" if result.created else "already_exists"
        }, message="Application created" if result.created else "Application already exists")

    def _run(self, *args, **kwargs) -> str:
        return "This tool requires async execution. Use _arun method."


class ChangeApplicationStageInput(BaseModel):
    application_id: str = Field(description="UUID of the application")
    stage_slug: str = Field(description="Slug of the target stage template (global)")


class ChangeApplicationStageTool(BaseTool):
    name: str = "change_application_stage"
    description: str = (
        "Move application to a new stage (free-form; logs history and updates status). Aliases accepted: under-review → in_review, in-review → in_review.\n\n"
        "Example: {\"application_id\": \"APP_UUID\", \"stage_slug\": \"under-review\"}"
    )
    args_schema: type[BaseModel] = ChangeApplicationStageInput

    async def _arun(
        self,
        application_id: str,
        stage_slug: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if ApplicationService is None:
            return _err("unavailable", "Applications service unavailable. Ensure migrations are applied.")
        service = ApplicationService()
        from asgiref.sync import sync_to_async
        try:
            app = await sync_to_async(service.change_stage)(application_id, stage_slug)
            return _ok({
                "application_id": str(app.id),
                "status": app.status,
                "current_stage": app.current_stage_instance.stage_template.slug if app.current_stage_instance else None,
            }, message="Stage changed")
        except Exception as e:
            # Suggest stages if slug invalid or not seeded
            from apps.applications.models import StageTemplate
            def list_slugs():
                return [s.slug for s in StageTemplate.objects.order_by("order").all()]
            slugs = await sync_to_async(list_slugs)()
            return _err(
                "change_failed",
                f"Failed to change stage: {str(e)}",
                hints=["Ensure stages are seeded via seed_default_stages", "Use one of the available slugs"],
                suggestions={"available_slugs": slugs, "aliases": {"under-review": "in_review", "in-review": "in_review"}},
            )

    def _run(self, *args, **kwargs) -> str:
        return "This tool requires async execution. Use _arun method."


class RecordApplicationDecisionInput(BaseModel):
    application_id: str = Field(description="UUID of the application")
    status: str = Field(description="Decision status: hired|rejected|offer")
    reason: Optional[str] = Field(default=None, description="Optional free-text reason")


class RecordApplicationDecisionTool(BaseTool):
    name: str = "decision_application"
    description: str = "Record a hiring decision (hired/rejected/offer) with optional reason."
    args_schema: type[BaseModel] = RecordApplicationDecisionInput

    async def _arun(
        self,
        application_id: str,
        status: str,
        reason: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if ApplicationService is None:
            return _err("unavailable", "Applications service unavailable. Ensure migrations are applied.")
        service = ApplicationService()
        from asgiref.sync import sync_to_async
        app = await sync_to_async(service.record_decision)(application_id, status, reason)
        return _ok({
            "application_id": str(app.id),
            "status": app.status,
        }, message="Decision recorded")

    def _run(self, *args, **kwargs) -> str:
        return "This tool requires async execution. Use _arun method."


# Update input schema to accept flat fields or nested question
class UpsertScreeningQuestionInput(BaseModel):
    opportunity_id: str = Field(description="UUID of the opportunity")
    # Either provide full question dict OR flat fields below
    question: Optional[Dict[str, Any]] = Field(default=None, description="Full question payload (preferred)")
    id: Optional[str] = Field(default=None, description="Existing question ID (for update)")
    question_text: Optional[str] = Field(default=None)
    question_type: Optional[str] = Field(default=None, description="text|boolean|single_choice|multi_choice|number")
    is_required: Optional[bool] = Field(default=None)
    order: Optional[int] = Field(default=None)
    config: Optional[Dict[str, Any]] = Field(default=None)


class UpsertScreeningQuestionTool(BaseTool):
    name: str = "upsert_screening_question"
    description: str = (
        "Create or update a screening question for an opportunity. Accepts either a nested 'question' object or flat fields.\n\n"
        "Examples (nested):\n"
        "{\"opportunity_id\": \"OPP_UUID\", \"question\": {\"question_text\": \"Visa needed?\", \"question_type\": \"boolean\", \"is_required\": false, \"order\": 1, \"config\": {\"disqualify_when\": true}}}\n\n"
        "Examples (flat):\n"
        "{\"opportunity_id\": \"OPP_UUID\", \"question_text\": \"Years exp?\", \"question_type\": \"number\", \"is_required\": true, \"order\": 2, \"config\": {\"min\": 3}}"
    )
    args_schema: type[BaseModel] = UpsertScreeningQuestionInput

    async def _arun(
        self,
        opportunity_id: str,
        question: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        question_text: Optional[str] = None,
        question_type: Optional[str] = None,
        is_required: Optional[bool] = None,
        order: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if ApplicationService is None:
            return _err("unavailable", "Applications service unavailable. Ensure migrations are applied.")
        # Build payload forgivingly
        q = dict(question) if isinstance(question, dict) else {}
        # Flat fields take precedence if provided
        if id is not None:
            q["id"] = id
        if question_text is not None:
            q["question_text"] = question_text
        if question_type is not None:
            q["question_type"] = question_type
        if is_required is not None:
            q["is_required"] = is_required
        if order is not None:
            q["order"] = order
        if config is not None:
            q["config"] = config

        # Normalize type aliases
        if "question_type" in q and isinstance(q["question_type"], str):
            qt = q["question_type"].strip().lower().replace("-", "_")
            alias = {
                "single": "single_choice",
                "multi": "multi_choice",
                "numeric": "number",
                "int": "number",
            }
            q["question_type"] = alias.get(qt, qt)

        # Validate required fields for create
        missing = []
        if not q.get("id") and not q.get("question_text"):
            missing.append("question_text")
        if not q.get("id") and not q.get("question_type"):
            missing.append("question_type")
        if missing:
            return _err(
                "missing_field",
                "Missing required fields to create question",
                fields=missing,
                hints=["Provide nested 'question' or flat fields (question_text, question_type)", "See examples in description"],
                suggestions={
                    "allowed_values": {"question_type": ["text","boolean","single_choice","multi_choice","number"]},
                    "example_flat": {
                        "opportunity_id": opportunity_id,
                        "question_text": "Why you?",
                        "question_type": "text",
                        "is_required": True,
                        "order": 1
                    }
                }
            )

        from asgiref.sync import sync_to_async
        service = ApplicationService()
        try:
            saved = await sync_to_async(service.upsert_screening_question)(opportunity_id, q)
        except Exception as e:
            return _err("upsert_failed", f"Failed to save question: {str(e)}")

        return _ok({
            "id": str(saved.id),
            "question_text": saved.question_text,
            "question_type": saved.question_type,
            "is_required": saved.is_required,
            "order": saved.order,
            "config": saved.config,
        }, message="Question saved")

    def _run(self, *args, **kwargs) -> str:
        return "This tool requires async execution. Use _arun method."


class DeleteScreeningQuestionInput(BaseModel):
    question_id: str = Field(description="UUID of the question to delete")


class DeleteScreeningQuestionTool(BaseTool):
    name: str = "delete_screening_question"
    description: str = "Delete a screening question."
    args_schema: type[BaseModel] = DeleteScreeningQuestionInput

    async def _arun(
        self,
        question_id: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if ApplicationService is None:
            return _err("unavailable", "Applications service unavailable. Ensure migrations are applied.")
        service = ApplicationService()
        from asgiref.sync import sync_to_async
        try:
            await sync_to_async(service.delete_screening_question)(question_id)
            return _ok({"question_id": question_id}, message="Question deleted")
        except Exception as e:
            return _err("delete_failed", f"Failed to delete question: {str(e)}")

    def _run(self, *args, **kwargs) -> str:
        return "This tool requires async execution. Use _arun method."


class SubmitApplicationAnswersInput(BaseModel):
    application_id: str = Field(description="UUID of the application")
    answers: List[Dict[str, Any]] = Field(description="Answers payload")


class SubmitApplicationAnswersTool(BaseTool):
    name: str = "submit_application_answers"
    description: str = (
        "Submit or update screening answers for an application. Provide answer_text for text/boolean/number, or answer_options for choices.\n\n"
        "Examples:\n"
        "{\"application_id\": \"APP_UUID\", \"answers\": [{\"question_id\": \"Q1\", \"answer_text\": true}, {\"question_id\": \"Q2\", \"answer_options\": [\"Citizen/PR\"]}]}"
    )
    args_schema: type[BaseModel] = SubmitApplicationAnswersInput

    async def _arun(
        self,
        application_id: str,
        answers: List[Dict[str, Any]],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if ApplicationService is None:
            return _err("unavailable", "Applications service unavailable. Ensure migrations are applied.")
        # Coerce common types
        def _coerce(v):
            if isinstance(v, str):
                vl = v.strip().lower()
                if vl in ("true", "yes"): return True
                if vl in ("false", "no"): return False
            return v
        coerced = []
        for a in answers:
            aa = dict(a)
            if "answer_text" in aa:
                aa["answer_text"] = _coerce(aa["answer_text"])  # bool/number strings tolerated
            coerced.append(aa)

        from asgiref.sync import sync_to_async
        service = ApplicationService()
        try:
            objs = await sync_to_async(service.submit_answers)(application_id, coerced)
        except Exception as e:
            return _err("submit_failed", f"Failed to submit answers: {str(e)}")

        return _ok({
            "application_id": application_id,
            "count": len(objs),
            "answers": [
                {
                    "id": str(o.id),
                    "question_id": str(o.question_id),
                    "is_disqualifying": o.is_disqualifying,
                } for o in objs
            ]
        }, message="Answers submitted")

    def _run(self, *args, **kwargs) -> str:
        return "This tool requires async execution. Use _arun method."


# Register ATS tools into role collections
try:
    EMPLOYER_TOOLS.extend([
        UpsertScreeningQuestionTool(),
        DeleteScreeningQuestionTool(),
        ChangeApplicationStageTool(),
        RecordApplicationDecisionTool(),
    ])
    CANDIDATE_TOOLS.extend([
        CreateApplicationTool(),
        SubmitApplicationAnswersTool(),
    ])
    # Super/general agent gets both via ALL_AGENT_TOOLS
    ALL_AGENT_TOOLS[:] = EMPLOYER_TOOLS + CANDIDATE_TOOLS
except Exception:
    pass


# Zero-arg schema for tools that take no input
class _EmptyArgs(BaseModel):
    pass

class SeedStagesTool(BaseTool):
    name: str = "seed_default_stages"
    description: str = "Create global default stages if missing (applied, in_review, interview, offer, hired, rejected, withdrawn)."
    args_schema: type[BaseModel] = _EmptyArgs

    async def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if ApplicationService is None:
            return _err("unavailable", "Applications service unavailable. Ensure migrations are applied.")
        service = ApplicationService()
        from asgiref.sync import sync_to_async
        try:
            stages = await sync_to_async(service.seed_default_stages)()
            return _ok({
                "count": len(stages),
                "slugs": [s.slug for s in stages]
            }, message="Seeded/ensured stages")
        except Exception as e:
            return _err("seed_failed", f"Failed to seed stages: {str(e)}")

    def _run(self, *args, **kwargs) -> str:
        return "This tool requires async execution. Use _arun method."


class ListStagesTool(BaseTool):
    name: str = "list_stages"
    description: str = "List global stage templates in order."
    args_schema: type[BaseModel] = _EmptyArgs

    async def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if ApplicationService is None:
            return _err("unavailable", "Applications service unavailable. Ensure migrations are applied.")
        service = ApplicationService()
        from asgiref.sync import sync_to_async
        try:
            stages = await sync_to_async(service.list_stages)()
            return _ok({
                "count": len(stages),
                "stages": [
                    {
                        "id": str(s.id),
                        "slug": s.slug,
                        "name": s.name,
                        "order": s.order,
                        "is_terminal": s.is_terminal,
                    } for s in stages
                ]
            }, message="Stages listed")
        except Exception as e:
            return _err("list_failed", f"Failed to list stages: {str(e)}")

    def _run(self, *args, **kwargs) -> str:
        return "This tool requires async execution. Use _arun method."


class UpsertStageInput(BaseModel):
    id: Optional[str] = Field(default=None, description="Stage ID to update (optional)")
    slug: Optional[str] = Field(default=None, description="Stage slug (required if creating)")
    name: Optional[str] = Field(default=None, description="Display name")
    order: Optional[int] = Field(default=None, description="Ordering value")
    is_terminal: Optional[bool] = Field(default=None, description="Terminal stage flag")


class UpsertStageTool(BaseTool):
    name: str = "upsert_stage"
    description: str = "Create or update a stage template by id or slug."
    args_schema: type[BaseModel] = UpsertStageInput

    async def _arun(
        self,
        id: Optional[str] = None,
        slug: Optional[str] = None,
        name: Optional[str] = None,
        order: Optional[int] = None,
        is_terminal: Optional[bool] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if ApplicationService is None:
            return _err("unavailable", "Applications service unavailable. Ensure migrations are applied.")
        service = ApplicationService()
        from asgiref.sync import sync_to_async
        try:
            stage = await sync_to_async(service.upsert_stage_template)({
                "id": id, "slug": slug, "name": name, "order": order, "is_terminal": is_terminal
            })
            return _ok({
                "id": str(stage.id),
                "slug": stage.slug,
                "name": stage.name,
                "order": stage.order,
                "is_terminal": stage.is_terminal,
            }, message="Stage saved")
        except Exception as e:
            return _err("upsert_failed", f"Failed to save stage: {str(e)}")

    def _run(self, *args, **kwargs) -> str:
        return "This tool requires async execution. Use _arun method."


class DeleteStageInput(BaseModel):
    identifier: str = Field(description="Stage id (UUID) or slug to delete")


class DeleteStageTool(BaseTool):
    name: str = "delete_stage"
    description: str = "Delete a stage by id or slug."
    args_schema: type[BaseModel] = DeleteStageInput

    async def _arun(
        self,
        identifier: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if ApplicationService is None:
            return _err("unavailable", "Applications service unavailable. Ensure migrations are applied.")
        service = ApplicationService()
        from asgiref.sync import sync_to_async
        try:
            count = await sync_to_async(service.delete_stage_template)(identifier)
            return _ok({"count": count}, message="Stages deleted")
        except Exception as e:
            return _err("delete_failed", f"Failed to delete stages: {str(e)}")

    def _run(self, *args, **kwargs) -> str:
        return "This tool requires async execution. Use _arun method."


class ScheduleInterviewInput(BaseModel):
    application_id: str = Field(description="UUID of the application")
    round_name: str = Field(description="Interview round name")
    scheduled_start: str = Field(description="ISO datetime, e.g. 2025-08-09T10:00:00Z or without Z in local time")
    scheduled_end: str = Field(description="ISO datetime, e.g. 2025-08-09T10:30:00Z")
    location_type: Optional[str] = Field(default="virtual", description="virtual|onsite")
    location_details: Optional[str] = Field(default="", description="Meeting link or address")


class ScheduleInterviewTool(BaseTool):
    name: str = "schedule_interview_minimal"
    description: str = (
        "Schedule a minimal interview for an application (round name, ISO datetimes, virtual/onsite).\n\n"
        "Example: {\"application_id\": \"APP_UUID\", \"round_name\": \"Initial\", \"scheduled_start\": \"2025-08-09T10:00:00Z\", \"scheduled_end\": \"2025-08-09T10:30:00Z\", \"location_type\": \"virtual\"}"
    )
    args_schema: type[BaseModel] = ScheduleInterviewInput

    async def _arun(
        self,
        application_id: str,
        round_name: str,
        scheduled_start: str,
        scheduled_end: str,
        location_type: str = "virtual",
        location_details: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if ApplicationService is None:
            return _err("unavailable", "Applications service unavailable. Ensure migrations are applied.")
        from datetime import datetime
        def parse_iso(s: str) -> datetime:
            return datetime.fromisoformat(s.replace("Z", "+00:00")) if isinstance(s, str) else s
        try:
            start_dt = parse_iso(scheduled_start)
            end_dt = parse_iso(scheduled_end)
        except Exception:
            return _err(
                "invalid_datetime",
                "scheduled_start and scheduled_end must be ISO datetimes",
                hints=["Use e.g. 2025-08-09T10:00:00Z"],
            )
        service = ApplicationService()
        from asgiref.sync import sync_to_async
        try:
            interview = await sync_to_async(service.schedule_interview_minimal)(
                application_id, round_name, start_dt, end_dt, location_type, location_details
            )
        except Exception as e:
            return _err("schedule_failed", f"Failed to schedule interview: {str(e)}")
        return _ok({
            "interview_id": str(interview.id),
            "application_id": str(interview.application_id),
            "round_name": interview.round_name,
            "scheduled_start": interview.scheduled_start.isoformat(),
            "scheduled_end": interview.scheduled_end.isoformat(),
            "location_type": interview.location_type,
        }, message="Interview scheduled")


class ListScreeningQuestionsInput(BaseModel):
    opportunity_id: str = Field(description="UUID of the opportunity")


class ListScreeningQuestionsTool(BaseTool):
    name: str = "list_screening_questions"
    description: str = "List screening questions for an opportunity."
    args_schema: type[BaseModel] = ListScreeningQuestionsInput

    async def _arun(
        self,
        opportunity_id: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        from asgiref.sync import sync_to_async
        from apps.applications.models import OpportunityQuestion
        def fetch(opp_id: str):
            return list(
                OpportunityQuestion.objects.filter(opportunity_id=opp_id).order_by("order").values(
                    "id", "question_text", "question_type", "is_required", "order", "config"
                )
            )
        rows = await sync_to_async(fetch)(opportunity_id)
        if not rows:
            return _ok({"opportunity_id": opportunity_id, "message": "No screening questions for this opportunity."})
        lines = [f"{r['id']}: {r['question_type']} | required={r['is_required']} | {r['question_text']}" for r in rows]
        return _ok({"opportunity_id": opportunity_id, "questions": lines})

    def _run(self, *args, **kwargs) -> str:
        return "This tool requires async execution. Use _arun method."


try:
    EMPLOYER_TOOLS.extend([
        SeedStagesTool(),
        ListStagesTool(),
        UpsertStageTool(),
        DeleteStageTool(),
        ScheduleInterviewTool(),
        ListScreeningQuestionsTool(),
    ])
    ALL_AGENT_TOOLS[:] = EMPLOYER_TOOLS + CANDIDATE_TOOLS
except Exception:
    pass