from django.db import models
from apps.organisations.models import Organisation
from apps.skills.models import Skill
import uuid


class Opportunity(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    organisation = models.ForeignKey(Organisation, on_delete=models.CASCADE, related_name='opportunities')
    
    title = models.CharField(max_length=255)
    description = models.TextField()
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.title} at {self.organisation}"


class OpportunitySkill(models.Model):
    """
    Junction model linking Opportunity to required/preferred Skills
    """
    REQUIREMENT_CHOICES = [
        ('required', 'Required'),
        ('preferred', 'Preferred'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    opportunity = models.ForeignKey(Opportunity, on_delete=models.CASCADE, related_name='opportunity_skills')
    skill = models.ForeignKey(Skill, on_delete=models.CASCADE, related_name='opportunity_skills')
    
    requirement_type = models.CharField(max_length=20, choices=REQUIREMENT_CHOICES, default='required')
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['opportunity', 'skill']
    
    def __str__(self):
        return f"{self.opportunity} - {self.skill} ({self.requirement_type})"


class OpportunityExperience(models.Model):
    """
    Experience requirements/preferences for an opportunity.
    Description field can be embedded for semantic matching.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    opportunity = models.ForeignKey(Opportunity, on_delete=models.CASCADE, related_name='opportunity_experiences')
    
    description = models.TextField()
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.opportunity} - {self.description[:50]}..."
