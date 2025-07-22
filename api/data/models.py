"""
Pydantic models for TalentCo data validation and serialization.
"""
from pydantic import BaseModel, Field, EmailStr, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class CareerLevel(str, Enum):
    """Career level enumeration."""
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    EXECUTIVE = "executive"


class EmploymentType(str, Enum):
    """Employment type enumeration."""
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    FREELANCE = "freelance"
    INTERNSHIP = "internship"


class WorkLocation(str, Enum):
    """Work location preference enumeration."""
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"


class UserProfile(BaseModel):
    """User profile model."""
    
    name: str = Field(..., description="User's full name")
    email: EmailStr = Field(..., description="User's email address")
    location: Optional[str] = Field(None, description="User's location")
    career_level: Optional[CareerLevel] = Field(None, description="User's career level")
    years_experience: Optional[int] = Field(None, ge=0, le=50, description="Years of experience")
    
    # Career preferences
    preferred_roles: List[str] = Field(default_factory=list, description="Preferred job roles")
    preferred_industries: List[str] = Field(default_factory=list, description="Preferred industries")
    work_location_preference: Optional[WorkLocation] = Field(None, description="Work location preference")
    employment_type_preference: Optional[EmploymentType] = Field(None, description="Employment type preference")
    
    # Skills and expertise
    skills: List[str] = Field(default_factory=list, description="User's skills")
    certifications: List[str] = Field(default_factory=list, description="User's certifications")
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional user metadata")
    
    @validator('years_experience')
    def validate_experience(cls, v):
        if v is not None and v < 0:
            raise ValueError('Years of experience cannot be negative')
        return v


class JobPost(BaseModel):
    """Job posting model."""
    
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    location: str = Field(..., description="Job location")
    description: str = Field(..., description="Job description")
    
    # Job details
    career_level: Optional[CareerLevel] = Field(None, description="Required career level")
    employment_type: EmploymentType = Field(..., description="Employment type")
    work_location: WorkLocation = Field(..., description="Work location type")
    
    # Requirements
    required_skills: List[str] = Field(default_factory=list, description="Required skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Preferred skills")
    min_experience: Optional[int] = Field(None, ge=0, description="Minimum years of experience")
    max_experience: Optional[int] = Field(None, ge=0, description="Maximum years of experience")
    
    # Compensation
    salary_min: Optional[int] = Field(None, ge=0, description="Minimum salary")
    salary_max: Optional[int] = Field(None, ge=0, description="Maximum salary")
    currency: Optional[str] = Field("USD", description="Salary currency")
    
    # Metadata
    posted_date: Optional[datetime] = Field(None, description="Job posting date")
    application_deadline: Optional[datetime] = Field(None, description="Application deadline")
    external_url: Optional[str] = Field(None, description="External job posting URL")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional job metadata")
    
    @validator('max_experience')
    def validate_experience_range(cls, v, values):
        if v is not None and 'min_experience' in values and values['min_experience'] is not None:
            if v < values['min_experience']:
                raise ValueError('Maximum experience cannot be less than minimum experience')
        return v
    
    @validator('salary_max')
    def validate_salary_range(cls, v, values):
        if v is not None and 'salary_min' in values and values['salary_min'] is not None:
            if v < values['salary_min']:
                raise ValueError('Maximum salary cannot be less than minimum salary')
        return v


class MatchResult(BaseModel):
    """Job match result model."""
    
    user_profile: UserProfile = Field(..., description="User profile")
    job_post: JobPost = Field(..., description="Job posting")
    
    # Match scores
    overall_score: float = Field(..., ge=0, le=1, description="Overall match score (0-1)")
    skill_score: float = Field(..., ge=0, le=1, description="Skills match score (0-1)")
    experience_score: float = Field(..., ge=0, le=1, description="Experience match score (0-1)")
    location_score: float = Field(..., ge=0, le=1, description="Location match score (0-1)")
    
    # Match details
    matching_skills: List[str] = Field(default_factory=list, description="Matching skills")
    missing_skills: List[str] = Field(default_factory=list, description="Missing skills")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    
    # Metadata
    match_timestamp: datetime = Field(default_factory=datetime.now, description="Match calculation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional match metadata")


class AgentResponse(BaseModel):
    """Agent response model."""
    
    message: str = Field(..., description="Agent response message")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Response confidence score")
    action_taken: Optional[str] = Field(None, description="Action taken by agent")
    memory_stored: bool = Field(False, description="Whether information was stored in memory")
    memory_retrieved: bool = Field(False, description="Whether information was retrieved from memory")
    
    # Structured data
    job_matches: List[MatchResult] = Field(default_factory=list, description="Job matches found")
    recommendations: List[str] = Field(default_factory=list, description="Career recommendations")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    session_id: Optional[str] = Field(None, description="Session identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata") 