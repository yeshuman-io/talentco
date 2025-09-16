# TalentCo Data Model & Evaluation System - Progress Report

## 📊 Project Overview

TalentCo is an AI-powered talent matching platform with bidirectional evaluation:
- **Employer Agent**: Co-creates job opportunities, finds and ranks candidates
- **Talent Agent**: Co-creates CVs, finds and ranks job opportunities  
- **Hybrid Matching**: Combines structured matching, semantic similarity, and LLM-as-Judge evaluation

---

## ✅ Completed Work

### 🏗️ **Data Model Architecture**

#### **Domain-Driven App Structure**
- **`profiles/`** - Talent domain (Profile, ProfileExperience, ProfileSkill, ProfileExperienceSkill)
- **`opportunities/`** - Job domain (Opportunity, OpportunitySkill, OpportunityExperience)  
- **`organisations/`** - Company domain (Organisation)
- **`skills/`** - Skills taxonomy (Skill)
- **`evaluations/`** - Matching/scoring domain (EvaluationSet, Evaluation)
- **`memories/`** - AI memory system (existing Memory models)

#### **Key Models Created**

**Profile Models** (`profiles/`)
```python
Profile - Person info (name, email, location, visa status)
ProfileExperience - Work history linked to organisations  
ProfileSkill - Skills with evidence levels (stated/experienced/evidenced)
ProfileExperienceSkill - Skills demonstrated in specific roles
```

**Opportunity Models** (`opportunities/`)
```python
Opportunity - Job postings linked to organisations
OpportunitySkill - Required/preferred skills for jobs
OpportunityExperience - Experience requirements (text for embedding)
```

**Evaluation Models** (`evaluations/`)
```python
EvaluationSet - Search session (employer or candidate perspective)
Evaluation - Individual match results with rankings and LLM reasoning
```

### 🔧 **Evaluation Service Architecture**

#### **Multi-Stage Pipeline**
1. **Structured Matching** - Skills overlap analysis (fast, cheap)
2. **Semantic Matching** - Vector similarity scoring (contextual)  
3. **Combined Scoring** - Weighted combination (60% structured, 40% semantic)
4. **LLM Judge** - Top 20% get expensive LLM evaluation with reasoning

#### **Cost-Controlled Design**
- **50 synthetic candidates** get structured + semantic evaluation
- **Top 10 candidates** (20%) get LLM judge evaluation
- **Configurable thresholds** for different use cases
- **Atomic transactions** ensure data consistency

#### **Bidirectional Support**
- **Employer POV**: `find_candidates_for_opportunity(opportunity_id)`
- **Candidate POV**: `find_opportunities_for_candidate(profile_id)`
- Same pipeline logic, different perspectives stored

### 🧹 **Code Organization**
- ✅ Removed unused admin.py, tests.py, views.py files from all apps
- ✅ Consolidated Experience models into profiles app  
- ✅ Moved junction models to appropriate domain apps
- ✅ Renamed Experience → ProfileExperience for consistency
- ✅ Clean import structure with no circular dependencies

---

## 🚧 Next Steps

### **Phase 1: Core Implementation** 

#### 1. **Add Vector Embeddings** 
```python
# Add to models:
profile_embedding = VectorField(dimensions=1536)  # Profile summary
skills_embedding = VectorField(dimensions=1536)   # Skills aggregation
opportunity_embedding = VectorField(dimensions=1536)  # Job description
requirements_embedding = VectorField(dimensions=1536)  # Requirements
```

#### 2. **Create Synthetic Data**
- Generate 50 diverse profiles with realistic skills/experience
- Create 20-30 job opportunities across different roles/industries
- Populate skills taxonomy with common technical/business skills
- Add organisations with various sizes/industries

#### 3. **Implement Real LLM Integration**
- Replace placeholder `_llm_judge_evaluation()` method
- Create prompts for profile-opportunity evaluation
- Parse LLM responses for scores and reasoning
- Add error handling and retry logic

#### 4. **Embedding Generation Pipeline**
- Create service for generating embeddings on model save
- Implement semantic similarity calculation using cosine similarity
- Add background task processing for expensive embedding operations
- Create embedding update triggers when related data changes

### **Phase 2: Agent Integration**

#### 5. **LangGraph Agent Tools**
- Create tool for `find_candidates_for_opportunity` 
- Create tool for `find_opportunities_for_candidate`
- Add tools for creating/updating profiles and opportunities
- Integrate with existing memory system for context

#### 6. **API Layer** 
- Django REST API endpoints for evaluation services
- Real-time evaluation status and progress tracking
- Webhook support for completed evaluations
- Rate limiting and authentication

### **Phase 3: Advanced Features**

#### 7. **Memory Integration**
- Store evaluation preferences and feedback in Memory system
- Learn from user interactions to improve future matching
- Track "why this worked/didn't work" patterns
- Personalized scoring weights based on user behavior

#### 8. **Evaluation Analytics**
- Evaluation success metrics and A/B testing
- Pipeline performance monitoring (structured vs semantic vs LLM effectiveness)
- Cost tracking and optimization suggestions
- User satisfaction correlation analysis

---

## 🎯 Success Metrics

### **POC Goals**
- [ ] Generate 50 synthetic profiles and 30 opportunities
- [ ] Successful bidirectional evaluation pipeline (employer + candidate POV)
- [ ] LLM judge working on top 10 candidates with reasoning
- [ ] Sub-30-second evaluation times for full pipeline
- [ ] Agent tools integrated with LangGraph

### **Production Readiness Goals**  
- [ ] Real-time embeddings generation (<5 seconds)
- [ ] Scalable evaluation processing (100+ candidates)
- [ ] Memory-driven personalization
- [ ] API response times <2 seconds
- [ ] 90%+ user satisfaction with match quality

---

## 🛠️ Technical Decisions Made

### **Architecture Choices**
- **EvaluationSet pattern**: Clean separation of search sessions vs individual results
- **Service layer**: Business logic separated from models for testability
- **Domain boundaries**: Each app owns its junction models for cohesion
- **Cost-first design**: Multi-stage pipeline minimizes expensive LLM calls

### **Data Modeling Decisions**
- **ProfileExperience naming**: Consistency with OpportunityExperience
- **Evidence levels**: Stated → Experienced → Evidenced skill progression
- **Flexible scoring**: JSON component_scores for pipeline extensibility
- **Bidirectional unique constraints**: Same pair can be evaluated from both perspectives

### **Embedding Strategy** 
- **Multiple embeddings per model**: Skills vs full profile vs requirements
- **Granular matching**: Specific embeddings (skills, experience) vs full document
- **Future-friendly**: VectorField dimensions standardized at 1536 (OpenAI compatible)

---

## 📝 Implementation Notes

- **Database**: PostgreSQL with pgvector extension required for embeddings
- **Dependencies**: Django, pgvector, python embedding libraries  
- **Memory usage**: Current design supports 50-100 profiles efficiently
- **Scaling considerations**: Background job processing needed for larger datasets

---

*Last Updated: December 2024*  
*Status: Data models complete, ready for implementation phase*