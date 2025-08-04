"""
Evaluation services for TalentCo matching algorithms.
Handles both employer->candidate and candidate->opportunity evaluations.
"""

from typing import List, Dict, Tuple
from django.utils import timezone
from django.db import transaction

from .models import EvaluationSet, Evaluation
from apps.profiles.models import Profile, ProfileSkill, ProfileExperience
from apps.opportunities.models import Opportunity, OpportunitySkill, OpportunityExperience
from apps.skills.models import Skill

# PostgreSQL vector similarity imports
from pgvector.django import CosineDistance
from django.db.models import Max, Case, When, F, Value, FloatField


class EvaluationService:
    """
    Service for creating and managing evaluation sets.
    Implements multi-stage pipeline: structured -> semantic -> LLM judge
    
    Pipeline:
    1. Structured matching (skills overlap, requirements)
    2. Semantic matching (PostgreSQL vector similarity) 
    3. LLM judge (only for matches above similarity threshold)
    """
    
    def __init__(self):
        pass  # Service is stateless
    
    def create_candidate_evaluation_set(
        self, 
        opportunity_id: str,
        llm_similarity_threshold: float = 0.7
    ) -> EvaluationSet:
        """
        Employer perspective: Find best candidates for an opportunity.
        
        Pipeline:
        1. Get all profiles 
        2. Structured matching (skills overlap, basic criteria)  
        3. Semantic matching (PostgreSQL vector similarity)
        4. Rank by combined score
        5. LLM judge only matches above similarity threshold
        """
        
        with transaction.atomic():
            # Get opportunity and all profiles
            opportunity = Opportunity.objects.get(id=opportunity_id)
            all_profiles = Profile.objects.all()
            
            # Create evaluation set
            eval_set = EvaluationSet.objects.create(
                evaluator_perspective='employer',
                opportunity=opportunity,
                total_evaluated=all_profiles.count(),
                llm_threshold_percent=llm_similarity_threshold  # Note: DB field name is historic, stores similarity threshold (not percentage)
            )
            
            # Run evaluation pipeline
            scored_profiles = self._evaluate_profiles_for_opportunity(
                all_profiles, opportunity
            )
            
            # Create evaluation records
            self._create_evaluation_records(
                eval_set, scored_profiles, llm_similarity_threshold, 'employer'
            )
            
            # Mark complete
            eval_set.is_complete = True
            eval_set.completed_at = timezone.now()
            eval_set.save()
            
            return eval_set
    
    def create_opportunity_evaluation_set(
        self, 
        profile_id: str,
        llm_similarity_threshold: float = 0.7
    ) -> EvaluationSet:
        """
        Candidate perspective: Find best opportunities for a profile.
        
        Pipeline:
        1. Get all opportunities
        2. Structured matching (skills overlap, basic criteria)  
        3. Semantic matching (PostgreSQL vector similarity)
        4. Rank by combined score
        5. LLM judge only matches above similarity threshold
        """
        
        with transaction.atomic():
            # Get profile and all opportunities  
            profile = Profile.objects.get(id=profile_id)
            all_opportunities = Opportunity.objects.all()
            
            # Create evaluation set
            eval_set = EvaluationSet.objects.create(
                evaluator_perspective='candidate',
                profile=profile,
                total_evaluated=all_opportunities.count(),
                llm_threshold_percent=llm_similarity_threshold  # Note: DB field name is historic, stores similarity threshold (not percentage)
            )
            
            # Run evaluation pipeline
            scored_opportunities = self._evaluate_opportunities_for_profile(
                all_opportunities, profile
            )
            
            # Create evaluation records (swap profile/opportunity order)
            self._create_evaluation_records(
                eval_set, scored_opportunities, llm_similarity_threshold, 'candidate'
            )
            
            # Mark complete
            eval_set.is_complete = True
            eval_set.completed_at = timezone.now()
            eval_set.save()
            
            return eval_set
    
    def _evaluate_profiles_for_opportunity(
        self, 
        profiles: List[Profile], 
        opportunity: Opportunity
    ) -> List[Dict]:
        """
        Run multi-stage evaluation pipeline for profiles against opportunity.
        """
        scored_profiles = []
        
        for profile in profiles:
            # Stage 1: Structured matching
            structured_score = self._calculate_structured_match(profile, opportunity)
            
            # Stage 2: Semantic matching  
            semantic_score = self._calculate_semantic_similarity(profile, opportunity)
            
            # Combined score (weighted)
            combined_score = (structured_score * 0.6) + (semantic_score * 0.4)
            
            scored_profiles.append({
                'profile': profile,
                'opportunity': opportunity,
                'structured_score': float(structured_score),  # Convert to Python float
                'semantic_score': float(semantic_score),      # Convert to Python float
                'combined_score': float(combined_score)       # Convert to Python float
            })
        
        # Sort by combined score (highest first)
        scored_profiles.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return scored_profiles
    
    def _evaluate_opportunities_for_profile(
        self, 
        opportunities: List[Opportunity], 
        profile: Profile
    ) -> List[Dict]:
        """
        Run multi-stage evaluation pipeline for opportunities against profile.
        Same logic, different perspective.
        """
        scored_opportunities = []
        
        for opportunity in opportunities:
            # Stage 1: Structured matching
            structured_score = self._calculate_structured_match(profile, opportunity)
            
            # Stage 2: Semantic matching
            semantic_score = self._calculate_semantic_similarity(profile, opportunity)
            
            # Combined score (weighted)
            combined_score = (structured_score * 0.6) + (semantic_score * 0.4)
            
            scored_opportunities.append({
                'profile': profile,
                'opportunity': opportunity,
                'structured_score': float(structured_score),  # Convert to Python float
                'semantic_score': float(semantic_score),      # Convert to Python float
                'combined_score': float(combined_score)       # Convert to Python float
            })
        
        # Sort by combined score (highest first)
        scored_opportunities.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return scored_opportunities
    
    def _create_evaluation_records(
        self,
        eval_set: EvaluationSet,
        scored_items: List[Dict],
        similarity_threshold: float,
        perspective: str
    ) -> None:
        """
        Create Evaluation records from scored results.
        Apply LLM judge only to matches above similarity threshold.
        """
        llm_judged_count = 0
        
        for rank, item in enumerate(scored_items, 1):
            # Check if combined score exceeds similarity threshold
            should_llm_judge = item['combined_score'] >= similarity_threshold
            
            # Create base evaluation
            evaluation = Evaluation.objects.create(
                evaluation_set=eval_set,
                profile=item['profile'],
                opportunity=item['opportunity'],
                final_score=item['combined_score'],
                rank_in_set=rank,
                component_scores={
                    'structured': item['structured_score'],
                    'semantic': item['semantic_score']
                },
                was_llm_judged=should_llm_judge
            )
            
            # Apply LLM judge only if similarity is high enough
            if should_llm_judge:
                llm_score, reasoning = self._llm_judge_evaluation(
                    item['profile'], item['opportunity'], perspective
                )
                
                # Update with LLM results
                evaluation.final_score = llm_score
                evaluation.llm_reasoning = reasoning
                evaluation.component_scores['llm_judge'] = llm_score
                evaluation.save()
        
                llm_judged_count += 1
        
        # Update evaluation set with actual LLM count
        eval_set.llm_judged_count = llm_judged_count
        eval_set.save()
    
    def _calculate_structured_match(self, profile: Profile, opportunity: Opportunity) -> float:
        """
        Calculate structured matching score based on skills overlap.
        Returns score between 0.0 and 1.0.
        """
        # Get profile skills
        profile_skills = set(
            ProfileSkill.objects.filter(profile=profile)
            .values_list('skill__name', flat=True)
        )
        
        # Get opportunity required skills
        required_skills = set(
            OpportunitySkill.objects.filter(
                opportunity=opportunity,
                requirement_type='required'
            ).values_list('skill__name', flat=True)
        )
        
        # Get opportunity preferred skills
        preferred_skills = set(
            OpportunitySkill.objects.filter(
                opportunity=opportunity,
                requirement_type='preferred'
            ).values_list('skill__name', flat=True)
        )
        
        # Calculate overlap scores
        if not required_skills and not preferred_skills:
            return 0.5  # No skills specified
        
        required_overlap = len(profile_skills & required_skills)
        preferred_overlap = len(profile_skills & preferred_skills)
        
        # Weighted scoring: required skills worth more
        required_score = required_overlap / len(required_skills) if required_skills else 1.0
        preferred_score = preferred_overlap / len(preferred_skills) if preferred_skills else 1.0
        
        # Combined score (required skills weighted 80%, preferred 20%)
        if required_skills:
            return (required_score * 0.8) + (preferred_score * 0.2)
        else:
            return preferred_score
    
    def _calculate_semantic_similarity(self, profile: Profile, opportunity: Opportunity) -> float:
        """
        Calculate multi-dimensional semantic similarity using PostgreSQL vector operations.
        
        Implements sophisticated matching strategy:
        1. Skills-to-skills matching (ProfileSkill ↔ OpportunitySkill)
        2. Experience-to-experience matching (ProfileExperience ↔ OpportunityExperience)  
        3. Skills-in-context matching (ProfileExperienceSkill ↔ OpportunitySkill) with temporal weighting
        4. Weighted combination of all dimensions
        
        Returns:
            Similarity score between 0.0 and 1.0
        """
        similarity_scores = {}
        
        # 1. Skills-to-Skills Matching (40% weight)
        skills_similarity = self._calculate_skills_similarity(profile, opportunity)
        similarity_scores['skills'] = skills_similarity
        
        # 2. Experience-to-Experience Matching (30% weight)  
        experience_similarity = self._calculate_experience_similarity(profile, opportunity)
        similarity_scores['experience'] = experience_similarity
        
        # 3. Skills-in-Context Matching with Temporal Weighting (30% weight)
        contextual_similarity = self._calculate_contextual_skills_similarity(profile, opportunity)
        similarity_scores['contextual'] = contextual_similarity
        
        # Weighted combination
        final_similarity = (
            (skills_similarity * 0.4) +
            (experience_similarity * 0.3) + 
            (contextual_similarity * 0.3)
        )
        
        return min(1.0, max(0.0, final_similarity))  # Clamp to [0, 1]
    
    def _ensure_list(self, embedding) -> List[float]:
        """Convert numpy array to list for pgvector compatibility"""
        if hasattr(embedding, 'tolist'):
            return embedding.tolist()
        elif isinstance(embedding, (list, tuple)):
            return list(embedding)
        else:
            return list(embedding)
    
    def _calculate_skills_similarity(self, profile: Profile, opportunity: Opportunity) -> float:
        """Calculate semantic similarity between ProfileSkills and OpportunitySkills using PostgreSQL"""
        
        profile_skills = profile.profile_skills.exclude(embedding__isnull=True)
        opportunity_skills = opportunity.opportunity_skills.exclude(embedding__isnull=True)
        
        if not profile_skills.exists() or not opportunity_skills.exists():
            return 0.0
        
        # For each opportunity skill, find the most similar profile skill using pgvector
        similarities = []
        
        for opp_skill in opportunity_skills:
            # Convert numpy array to list for pgvector compatibility
            opp_embedding = self._ensure_list(opp_skill.embedding)
            
            # Find the most similar profile skill using Django ORM + pgvector
            best_match = profile_skills.annotate(
                similarity=1 - CosineDistance('embedding', opp_embedding)
            ).aggregate(
                max_similarity=Max('similarity')
            )['max_similarity']
            
            similarities.append(best_match or 0.0)
        
        # Return average of best matches for each opportunity skill
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_experience_similarity(self, profile: Profile, opportunity: Opportunity) -> float:
        """Calculate semantic similarity between ProfileExperiences and OpportunityExperiences using PostgreSQL"""
        
        profile_experiences = profile.profile_experiences.exclude(embedding__isnull=True)
        opportunity_experiences = opportunity.opportunity_experiences.exclude(embedding__isnull=True)
        
        if not profile_experiences.exists() or not opportunity_experiences.exists():
            return 0.0
        
        # For each opportunity experience, find the most similar profile experience using pgvector
        similarities = []
        
        for opp_exp in opportunity_experiences:
            # Convert numpy array to list for pgvector compatibility
            opp_embedding = self._ensure_list(opp_exp.embedding)
            
            # Find the most similar profile experience using Django ORM + pgvector
            best_match = profile_experiences.annotate(
                similarity=1 - CosineDistance('embedding', opp_embedding)
            ).aggregate(
                max_similarity=Max('similarity')
            )['max_similarity']
            
            similarities.append(best_match or 0.0)
        
        # Return average of best matches for each opportunity experience
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_contextual_skills_similarity(self, profile: Profile, opportunity: Opportunity) -> float:
        """
        Calculate semantic similarity between ProfileExperienceSkills and OpportunitySkills
        with temporal weighting using PostgreSQL. This is the key innovation - skills demonstrated 
        in context with recency bias.
        """
        from apps.profiles.models import ProfileExperienceSkill
        
        opportunity_skills = opportunity.opportunity_skills.exclude(embedding__isnull=True)
        
        if not opportunity_skills.exists():
            return 0.0
        
        # Get all ProfileExperienceSkills for this profile with embeddings
        profile_exp_skills = ProfileExperienceSkill.objects.filter(
            profile_experience__profile=profile,
            embedding__isnull=False
        ).select_related('profile_experience')
        
        if not profile_exp_skills.exists():
            return 0.0
        
        # For each opportunity skill, find best matching experience skill with temporal weighting
        weighted_similarities = []
        
        for opp_skill in opportunity_skills:
            # Convert numpy array to list for pgvector compatibility
            opp_embedding = self._ensure_list(opp_skill.embedding)
            
            # Annotate experience skills with similarity and temporal weight using Django ORM
            annotated_exp_skills = profile_exp_skills.annotate(
                # Calculate cosine similarity using pgvector
                base_similarity=1 - CosineDistance('embedding', opp_embedding),
                
                # Simple temporal weight: current roles get 1.0, past roles get 0.7
                # This avoids complex date arithmetic that causes ORM issues
                temporal_weight=Case(
                    When(profile_experience__end_date__isnull=True, then=Value(1.0)),  # Current role
                    default=Value(0.7),  # Past role
                    output_field=FloatField()
                ),
                
                # Final weighted similarity
                weighted_similarity=F('base_similarity') * F('temporal_weight')
            )
            
            # Get the best weighted match for this opportunity skill
            best_weighted_match = annotated_exp_skills.aggregate(
                max_weighted=Max('weighted_similarity')
            )['max_weighted']
            
            weighted_similarities.append(best_weighted_match or 0.0)
        
        # Return average weighted similarity across all opportunity skills
        return sum(weighted_similarities) / len(weighted_similarities) if weighted_similarities else 0.0
    
    def _llm_judge_evaluation(
        self, 
        profile: Profile, 
        opportunity: Opportunity, 
        perspective: str
    ) -> Tuple[float, str]:
        """
        Use LLM to evaluate profile-opportunity match with detailed reasoning.
        Returns (score, reasoning).
        """
        import openai
        from django.conf import settings
        import json
        
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Format detailed profile data
        profile_skills = [ps.skill.name for ps in profile.profile_skills.all()]
        
        # Get detailed experience information
        experiences = []
        for exp in profile.profile_experiences.all().order_by('-end_date'):
            exp_text = f"• {exp.title} at {exp.organisation.name} ({exp.start_date.year}-{'Present' if not exp.end_date else exp.end_date.year})"
            if exp.description:
                exp_text += f": {exp.description[:200]}..."
            experiences.append(exp_text)
        
        # Format opportunity data with more detail
        required_skills = [os.skill.name for os in opportunity.opportunity_skills.filter(requirement_type='required')]
        preferred_skills = [os.skill.name for os in opportunity.opportunity_skills.filter(requirement_type='preferred')]
        
        # Get opportunity experience requirements
        opp_experiences = []
        for opp_exp in opportunity.opportunity_experiences.all():
            opp_experiences.append(f"• {opp_exp.description}")
        
        if perspective == 'employer':
            prompt = f"""You are an expert ESG talent consultant evaluating whether a candidate is a good fit for a sustainability role.

🎯 OPPORTUNITY: {opportunity.title} at {opportunity.organisation.name}
Description: {opportunity.description}

Required Skills: {', '.join(required_skills) if required_skills else 'None specified'}
Preferred Skills: {', '.join(preferred_skills) if preferred_skills else 'None specified'}

Experience Requirements:
{chr(10).join(opp_experiences) if opp_experiences else 'None specified'}

👤 CANDIDATE: {profile.first_name} {profile.last_name}
Skills Portfolio: {', '.join(profile_skills) if profile_skills else 'None listed'}

Career History:
{chr(10).join(experiences) if experiences else 'No experience listed'}

Provide a comprehensive evaluation analyzing:
1. Skills Match: How well do their skills align with requirements?
2. Experience Relevance: How relevant is their career background?
3. Growth Potential: What's their potential for this role?
4. ESG Sector Fit: How well do they fit the ESG/sustainability domain?
5. Overall Assessment: Strengths, gaps, and recommendation

Score from 0.0 to 1.0 where:
• 0.9-1.0: Exceptional match, immediate hire
• 0.7-0.9: Strong match, proceed to interview
• 0.5-0.7: Good potential, needs further evaluation
• 0.3-0.5: Some alignment, significant gaps
• 0.0-0.3: Poor match, not suitable

Respond in JSON format: {{"score": 0.85, "reasoning": "detailed analysis here"}}"""
        else:
            prompt = f"""You are an expert career consultant evaluating whether a sustainability role is a good fit for a candidate.

👤 CANDIDATE: {profile.first_name} {profile.last_name}
Skills Portfolio: {', '.join(profile_skills) if profile_skills else 'None listed'}

Career History:
{chr(10).join(experiences) if experiences else 'No experience listed'}

🎯 OPPORTUNITY: {opportunity.title} at {opportunity.organisation.name}
Description: {opportunity.description}

Required Skills: {', '.join(required_skills) if required_skills else 'None specified'}
Preferred Skills: {', '.join(preferred_skills) if preferred_skills else 'None specified'}

Experience Requirements:
{chr(10).join(opp_experiences) if opp_experiences else 'None specified'}

Provide a comprehensive career fit analysis covering:
1. Skill Development: How does this role advance their skills?
2. Career Progression: Is this a logical next step?
3. Learning Opportunities: What new capabilities will they gain?
4. Industry Alignment: How well does this fit their ESG interests?
5. Overall Recommendation: Why this is/isn't a good move

Score from 0.0 to 1.0 where:
• 0.9-1.0: Perfect career move, highly recommended
• 0.7-0.9: Excellent opportunity, should pursue
• 0.5-0.7: Good fit, worth considering
• 0.3-0.5: Mixed fit, some benefits but gaps
• 0.0-0.3: Poor fit, not recommended

Respond in JSON format: {{"score": 0.85, "reasoning": "detailed analysis here"}}"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert ESG talent consultant with deep knowledge of sustainability careers, environmental frameworks (TCFD, GRI, SASB), and ESG investment strategies. Provide detailed, actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)
            
            score = float(result.get('score', 0.0))
            reasoning = result.get('reasoning', 'LLM evaluation completed')
            
            # Clamp score to valid range
            score = max(0.0, min(1.0, score))
            
            return score, reasoning
            
        except Exception as e:
            # Fallback on API failure - return conservative score
            fallback_score = 0.7
            fallback_reasoning = f"LLM evaluation failed ({str(e)}), using conservative score"
            return fallback_score, fallback_reasoning


# Convenience functions for common use cases
def find_candidates_for_opportunity(opportunity_id: str, llm_similarity_threshold: float = 0.7) -> EvaluationSet:
    """Employer: Find best candidates for a job opportunity."""
    service = EvaluationService()
    return service.create_candidate_evaluation_set(opportunity_id, llm_similarity_threshold)


def find_opportunities_for_candidate(profile_id: str, llm_similarity_threshold: float = 0.7) -> EvaluationSet:
    """Candidate: Find best job opportunities for a profile."""
    service = EvaluationService()
    return service.create_opportunity_evaluation_set(profile_id, llm_similarity_threshold)


def get_top_matches(evaluation_set: EvaluationSet, limit: int = 10) -> List[Evaluation]:
    """Get top N matches from an evaluation set."""
    return evaluation_set.evaluations.order_by('rank_in_set')[:limit]


def get_llm_judged_matches(evaluation_set: EvaluationSet) -> List[Evaluation]:
    """Get only the matches that were evaluated by LLM."""
    return evaluation_set.evaluations.filter(was_llm_judged=True).order_by('rank_in_set')