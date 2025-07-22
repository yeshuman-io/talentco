FROM langchain/langgraph-api:3.11

# Set environment variables for uv
ENV UV_SYSTEM_PYTHON=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Node.js 22 LTS and npm for UI compilation using NodeSource (most reliable for mixed environments)
# ALSO install pgvector system dependencies
RUN apt-get update && \
    apt-get install -y ca-certificates curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    apt-get install -y --no-install-recommends \
        postgresql-client \
        libpq-dev \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    node --version && \
    npm --version && \
    npx --version && \
    rm -f /usr/lib/python3.11/EXTERNALLY-MANAGED /usr/lib/python3/dist-packages/EXTERNALLY-MANAGED

# Copy application code to expected location
COPY . /deps/api

# Set working directory
WORKDIR /deps/api

# Install dependencies to system Python (not virtual env)
RUN uv pip install --system --no-cache-dir .

# CRITICAL: Create stub licensing module to avoid circular import
RUN echo "def check_license_periodically(): pass" > /api/langgraph_license/validation.py

# Force our Python packages to take precedence over container's built-in packages
ENV PYTHONPATH="/usr/local/lib/python3.11/site-packages:/deps/api:$PYTHONPATH"

# Set the graph configuration for talentco agent
ENV LANGSERVE_GRAPHS='{"agent": "/deps/api/graphs/agent.py:graph"}'

# Set UI environment variables for production
ENV LANGGRAPH_UI='{"agent": "./graphs/agent.py"}'
ENV LANGGRAPH_UI_BUNDLER=true

# Set explicit NODE_PATH and PATH for UI bundler subprocess
ENV NODE_PATH=/usr/lib/node_modules
ENV PATH=/usr/bin:$PATH
# Ensure subprocess environment compatibility
ENV PYTHONUNBUFFERED=1
ENV SHELL=/bin/bash

# Create required UI bundler directories
RUN mkdir -p /deps/api/.langgraph_api/ui

# Pre-install common UI dependencies to avoid subprocess issues
RUN npm install -g typescript @types/react @types/react-dom || true

# Ensure langgraph-api components remain accessible
RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license && \
    touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py

# CRITICAL: Reinstall LangGraph API components (including UI) to prevent overwriting
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps -e /api

# Add our project to Python path
ENV PYTHONPATH="/deps/api:$PYTHONPATH"

# Allow pip system installs
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Patch Python runtime to ensure asyncio subprocess env is a dict (uvloop compatibility)
RUN echo "import asyncio, os" > /usr/local/lib/python3.11/site-packages/sitecustomize.py && \
    echo "" >> /usr/local/lib/python3.11/site-packages/sitecustomize.py && \
    echo "_orig_create = asyncio.create_subprocess_exec" >> /usr/local/lib/python3.11/site-packages/sitecustomize.py && \
    echo "" >> /usr/local/lib/python3.11/site-packages/sitecustomize.py && \
    echo "async def _fixed_create(*args, **kwargs):" >> /usr/local/lib/python3.11/site-packages/sitecustomize.py && \
    echo "    env = kwargs.get('env')" >> /usr/local/lib/python3.11/site-packages/sitecustomize.py && \
    echo "    if isinstance(env, os._Environ):" >> /usr/local/lib/python3.11/site-packages/sitecustomize.py && \
    echo "        kwargs['env'] = dict(env)" >> /usr/local/lib/python3.11/site-packages/sitecustomize.py && \
    echo "    return await _orig_create(*args, **kwargs)" >> /usr/local/lib/python3.11/site-packages/sitecustomize.py && \
    echo "" >> /usr/local/lib/python3.11/site-packages/sitecustomize.py && \
    echo "asyncio.create_subprocess_exec = _fixed_create" >> /usr/local/lib/python3.11/site-packages/sitecustomize.py 