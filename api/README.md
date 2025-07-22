# TalentCo API

Django backend with integrated LangGraph agent for career guidance and job matching in sustainability and ESG sectors.

## Architecture

This API follows the Yes Human Agentic Stack specification:

- **Django Framework**: Web framework with ORM and admin interface
- **LangGraph Integration**: ReAct agent with memory capabilities
- **Memory Backend**: Custom Django-based Mem0 implementation
- **PostgreSQL + pgvector**: Database with vector search capabilities

## Quick Start

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run migrations**:
   ```bash
   python manage.py migrate
   ```

4. **Start development servers**:
   ```bash
   # Django server
   python manage.py runserver

   # LangGraph dev server (in another terminal)
   langgraph dev
   ```

## Directory Structure

- `talentco/` - Django project settings
- `apps/` - Django applications
  - `memories/` - Memory storage backend
  - `core/` - Core models and utilities
  - `api/` - API endpoints
- `graphs/` - LangGraph agent definitions
- `tools/` - Agent tools and capabilities
- `clients/` - External API clients
- `data/` - Pydantic models and validation

## Agent Capabilities

The TalentCo agent specializes in:

- Career guidance for sustainability and ESG roles
- Job matching and recommendations
- Skills assessment and development planning
- Industry insights and trend analysis
- Interview preparation and resume guidance

## API Endpoints

- `GET /api/v1/health/` - Health check endpoint
- More endpoints will be added as the agent evolves

## Memory System

The agent uses a custom Django-based memory backend that:

- Stores career information and preferences
- Performs semantic search using vector embeddings
- Categorizes memories by type (experience, skills, preferences, goals)
- Tracks search analytics for continuous improvement

## Development

See the main project README for development guidelines and deployment instructions. 