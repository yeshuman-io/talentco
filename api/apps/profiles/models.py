from django.db import models
from apps.organisations.models import Organisation
from apps.skills.models import Skill
import uuid


class Profile(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.first_name} {self.last_name}"


class ProfileExperience(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE, related_name='profile_experiences')
    organisation = models.ForeignKey(Organisation, on_delete=models.CASCADE, related_name='profile_experiences')
    
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    start_date = models.DateField()
    end_date = models.DateField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.profile} - {self.title} at {self.organisation}"


class ProfileSkill(models.Model):
    """
    Junction model linking Profile to Skills with evidence level
    """
    EVIDENCE_CHOICES = [
        ('stated', 'Stated Competency'),
        ('experienced', 'Experience-based'),
        ('evidenced', 'Evidenced Competency'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE, related_name='profile_skills')
    skill = models.ForeignKey(Skill, on_delete=models.CASCADE, related_name='profile_skills')
    
    evidence_level = models.CharField(max_length=20, choices=EVIDENCE_CHOICES, default='stated')
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['profile', 'skill']
    
    def __str__(self):
        return f"{self.profile} - {self.skill} ({self.evidence_level})"


class ProfileExperienceSkill(models.Model):
    """
    Junction model linking ProfileExperience to Skills demonstrated in that role
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    profile_experience = models.ForeignKey(ProfileExperience, on_delete=models.CASCADE, related_name='profile_experience_skills')
    skill = models.ForeignKey(Skill, on_delete=models.CASCADE, related_name='profile_experience_skills')
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['profile_experience', 'skill']
    
    def __str__(self):
        return f"{self.profile_experience} - {self.skill}"
