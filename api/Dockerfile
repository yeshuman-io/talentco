FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for pgvector and general use
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    libpq-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first for better Docker layer caching
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Set environment variables for production
ENV DJANGO_SETTINGS_MODULE=talentco.settings
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8080

# Run the same commands that work locally
CMD ["sh", "-c", "python manage.py setup_pgvector && python manage.py migrate && uv run langgraph dev --host 0.0.0.0 --port $PORT"] 