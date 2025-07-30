"""
Evaluation services for TalentCo matching algorithms.
Handles both employer->candidate and candidate->opportunity evaluations.
"""

from typing import List, Dict, Tuple, Optional
from django.utils import timezone
from django.db import transaction

from .models import EvaluationSet, Evaluation
from apps.profiles.models import Profile, ProfileSkill, ProfileExperience
from apps.opportunities.models import Opportunity, OpportunitySkill, OpportunityExperience
from apps.skills.models import Skill


class EvaluationService:
    """
    Service for creating and managing evaluation sets.
    Implements multi-stage pipeline: structured -> semantic -> LLM judge
    """
    
    def __init__(self):
        self.llm_threshold_percent = 0.2  # Top 20% get LLM evaluation
    
    def create_candidate_evaluation_set(
        self, 
        opportunity_id: str,
        llm_threshold_percent: Optional[float] = None
    ) -> EvaluationSet:
        """
        Employer perspective: Find best candidates for an opportunity.
        
        Pipeline:
        1. Get all profiles (50 synthetic)
        2. Structured matching (skills overlap, basic criteria)  
        3. Semantic matching (vector similarity)
        4. Rank by combined score
        5. LLM judge top 20% (10 candidates)
        """
        threshold = llm_threshold_percent or self.llm_threshold_percent
        
        with transaction.atomic():
            # Get opportunity and all profiles
            opportunity = Opportunity.objects.get(id=opportunity_id)
            all_profiles = Profile.objects.all()
            
            # Create evaluation set
            eval_set = EvaluationSet.objects.create(
                evaluator_perspective='employer',
                opportunity=opportunity,
                total_evaluated=all_profiles.count(),
                llm_threshold_percent=threshold
            )
            
            # Run evaluation pipeline
            scored_profiles = self._evaluate_profiles_for_opportunity(
                all_profiles, opportunity
            )
            
            # Create evaluation records
            self._create_evaluation_records(
                eval_set, scored_profiles, threshold, 'employer'
            )
            
            # Mark complete
            eval_set.is_complete = True
            eval_set.completed_at = timezone.now()
            eval_set.save()
            
            return eval_set
    
    def create_opportunity_evaluation_set(
        self, 
        profile_id: str,
        llm_threshold_percent: Optional[float] = None
    ) -> EvaluationSet:
        """
        Candidate perspective: Find best opportunities for a profile.
        Same pipeline, reversed perspective.
        """
        threshold = llm_threshold_percent or self.llm_threshold_percent
        
        with transaction.atomic():
            # Get profile and all opportunities  
            profile = Profile.objects.get(id=profile_id)
            all_opportunities = Opportunity.objects.all()
            
            # Create evaluation set
            eval_set = EvaluationSet.objects.create(
                evaluator_perspective='candidate',
                profile=profile,
                total_evaluated=all_opportunities.count(),
                llm_threshold_percent=threshold
            )
            
            # Run evaluation pipeline
            scored_opportunities = self._evaluate_opportunities_for_profile(
                all_opportunities, profile
            )
            
            # Create evaluation records (swap profile/opportunity order)
            self._create_evaluation_records(
                eval_set, scored_opportunities, threshold, 'candidate'
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
                'structured_score': structured_score,
                'semantic_score': semantic_score,
                'combined_score': combined_score
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
                'structured_score': structured_score,
                'semantic_score': semantic_score,
                'combined_score': combined_score
            })
        
        # Sort by combined score (highest first)
        scored_opportunities.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return scored_opportunities
    
    def _create_evaluation_records(
        self,
        eval_set: EvaluationSet,
        scored_items: List[Dict],
        llm_threshold_percent: float,
        perspective: str
    ) -> None:
        """
        Create Evaluation records from scored results.
        Apply LLM judge to top performers only.
        """
        total_count = len(scored_items)
        llm_cut_count = max(1, int(total_count * llm_threshold_percent))
        
        for rank, item in enumerate(scored_items, 1):
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
                was_llm_judged=rank <= llm_cut_count
            )
            
            # Apply LLM judge to top performers
            if rank <= llm_cut_count:
                llm_score, reasoning = self._llm_judge_evaluation(
                    item['profile'], item['opportunity'], perspective
                )
                
                # Update with LLM results
                evaluation.final_score = llm_score
                evaluation.llm_reasoning = reasoning
                evaluation.component_scores['llm_judge'] = llm_score
                evaluation.save()
        
        # Update evaluation set with LLM count
        eval_set.llm_judged_count = llm_cut_count
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
        Calculate semantic similarity using vector embeddings.
        TODO: Implement once embeddings are added to models.
        """
        # Placeholder implementation
        # In reality, this would:
        # 1. Get profile embedding (skills + experience summary)
        # 2. Get opportunity embedding (description + requirements)
        # 3. Calculate cosine similarity between vectors
        # 4. Return normalized similarity score
        
        import random
        return random.uniform(0.3, 0.9)  # Fake similarity for now
    
    def _llm_judge_evaluation(
        self, 
        profile: Profile, 
        opportunity: Opportunity, 
        perspective: str
    ) -> Tuple[float, str]:
        """
        Use LLM to evaluate profile-opportunity match.
        Returns (score, reasoning).
        TODO: Implement LLM integration.
        """
        # Placeholder implementation
        # In reality, this would:
        # 1. Format profile and opportunity data for LLM
        # 2. Send to LLM with evaluation prompt
        # 3. Parse response for score and reasoning
        # 4. Return structured results
        
        import random
        score = random.uniform(0.6, 0.95)  # Top candidates get higher scores
        
        if perspective == 'employer':
            reasoning = f"Strong technical match for {opportunity.title}. Profile shows relevant experience and skills alignment."
        else:
            reasoning = f"Good opportunity fit for candidate. Role aligns with career goals and skill set."
        
        return score, reasoning


# Convenience functions for common use cases
def find_candidates_for_opportunity(opportunity_id: str, llm_threshold: float = 0.2) -> EvaluationSet:
    """Employer: Find best candidates for a job opportunity."""
    service = EvaluationService()
    return service.create_candidate_evaluation_set(opportunity_id, llm_threshold)


def find_opportunities_for_candidate(profile_id: str, llm_threshold: float = 0.2) -> EvaluationSet:
    """Candidate: Find best job opportunities for a profile."""
    service = EvaluationService()
    return service.create_opportunity_evaluation_set(profile_id, llm_threshold)


def get_top_matches(evaluation_set: EvaluationSet, limit: int = 10) -> List[Evaluation]:
    """Get top N matches from an evaluation set."""
    return evaluation_set.evaluations.order_by('rank_in_set')[:limit]


def get_llm_judged_matches(evaluation_set: EvaluationSet) -> List[Evaluation]:
    """Get only the matches that were evaluated by LLM."""
    return evaluation_set.evaluations.filter(was_llm_judged=True).order_by('rank_in_set')