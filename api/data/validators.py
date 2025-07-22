"""
Data validation utilities for TalentCo.
"""
import re
from typing import List, Optional, Any
from pydantic import validator


def validate_email(email: str) -> str:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValueError("Invalid email format")
    return email.lower()


def validate_skills_list(skills: List[str]) -> List[str]:
    """Validate and normalize skills list."""
    if not skills:
        return []
    
    # Remove duplicates and normalize
    normalized_skills = []
    seen = set()
    
    for skill in skills:
        if isinstance(skill, str):
            normalized = skill.strip().title()
            if normalized and normalized.lower() not in seen:
                normalized_skills.append(normalized)
                seen.add(normalized.lower())
    
    return normalized_skills


def validate_location(location: str) -> str:
    """Validate and normalize location string."""
    if not location or not isinstance(location, str):
        raise ValueError("Location must be a non-empty string")
    
    # Basic normalization
    normalized = location.strip().title()
    if not normalized:
        raise ValueError("Location cannot be empty")
    
    return normalized


def validate_salary_range(min_salary: Optional[int], max_salary: Optional[int]) -> tuple:
    """Validate salary range."""
    if min_salary is not None and min_salary < 0:
        raise ValueError("Minimum salary cannot be negative")
    
    if max_salary is not None and max_salary < 0:
        raise ValueError("Maximum salary cannot be negative")
    
    if min_salary is not None and max_salary is not None:
        if max_salary < min_salary:
            raise ValueError("Maximum salary cannot be less than minimum salary")
    
    return min_salary, max_salary


def validate_experience_years(years: Optional[int]) -> Optional[int]:
    """Validate years of experience."""
    if years is None:
        return None
    
    if not isinstance(years, int):
        raise ValueError("Years of experience must be an integer")
    
    if years < 0:
        raise ValueError("Years of experience cannot be negative")
    
    if years > 60:
        raise ValueError("Years of experience cannot exceed 60")
    
    return years


def normalize_job_title(title: str) -> str:
    """Normalize job title."""
    if not title or not isinstance(title, str):
        raise ValueError("Job title must be a non-empty string")
    
    # Basic normalization
    normalized = title.strip()
    if not normalized:
        raise ValueError("Job title cannot be empty")
    
    # Capitalize each word
    return ' '.join(word.capitalize() for word in normalized.split())


def validate_url(url: Optional[str]) -> Optional[str]:
    """Validate URL format."""
    if not url:
        return None
    
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if not url_pattern.match(url):
        raise ValueError("Invalid URL format")
    
    return url 