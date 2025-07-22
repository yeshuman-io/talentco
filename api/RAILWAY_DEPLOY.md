# Railway Deployment Guide for TalentCo API

## 🚂 What's Been Updated

### ✅ Changes Made:

1. **Dockerfile**: Added pgvector system dependencies (`postgresql-client`, `libpq-dev`, `build-essential`)
2. **railway.json**: Added `startCommand` to run pgvector setup and Django migrations on deploy
3. **Management Command**: Created `setup_pgvector.py` to enable pgvector extension

### 📦 Dependencies Already in Place:
- ✅ `pgvector>=0.3.2` (Python client)
- ✅ `psycopg2-binary>=2.9.9` (PostgreSQL adapter)
- ✅ `uv` for fast dependency management
- ✅ Django + LangGraph integration

## 🚀 Deploy to Railway

### 1. Deploy from CLI:
```bash
# From the api/ directory
railway deploy
```

### 2. Set Environment Variables in Railway:
```bash
# Required environment variables
railway variables set DJANGO_SECRET_KEY="your-secret-key-here"
railway variables set DEBUG="False"
railway variables set ALLOWED_HOSTS="your-domain.railway.app"
railway variables set OPENAI_API_KEY="your-openai-key"

# Optional LangSmith tracing
railway variables set LANGCHAIN_API_KEY="your-langsmith-key"
railway variables set LANGCHAIN_TRACING_V2="true"
railway variables set LANGCHAIN_PROJECT="talentco-production"
```

### 3. Database Setup:
Railway will automatically:
1. Provision a PostgreSQL database
2. Set the `DATABASE_URL` environment variable
3. Enable pgvector extension (via our management command)
4. Run Django migrations

## 🔧 What Happens on Deploy:

1. **Build**: `uv sync --frozen` (installs dependencies)
2. **Deploy**: 
   - `python manage.py setup_pgvector` (enables pgvector extension)
   - `python manage.py migrate` (runs Django migrations) 
   - `langgraph dev --host 0.0.0.0 --port $PORT` (starts the server)

## 🗄️ Database Configuration

Your Django settings already support:
- ✅ Environment-based `DATABASE_URL` (Railway auto-provides this)
- ✅ PostgreSQL with pgvector support
- ✅ Fallback configuration for local development

## 🎯 Access Your Deployed API

After deployment:
- **LangGraph API**: `https://your-app.railway.app/`
- **Django Admin**: `https://your-app.railway.app/admin/` (if enabled)
- **API Endpoints**: `https://your-app.railway.app/api/` (based on your Django URLs)

## 🔍 Troubleshooting

### If pgvector fails to install:
Railway's PostgreSQL should support pgvector by default. If you get permission errors:
1. The management command gracefully handles this
2. Deployment continues even if extension setup fails
3. pgvector might already be enabled on the database

### If migrations fail:
```bash
# Check logs in Railway dashboard
railway logs

# Or run migrations manually
railway run python manage.py migrate
```

### If deployment is slow:
- uv is much faster than pip
- Railway caches dependencies between deployments
- First deploy takes longer (subsequent deploys are faster)

## 📊 Monitoring

Railway provides:
- Real-time logs
- Resource usage metrics
- Auto-scaling (if needed)
- Zero-downtime deployments

Your Django + LangGraph + pgvector stack is now ready for production! 🎉 