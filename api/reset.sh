#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment variables from .env
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo -e "${RED}Error: .env file not found${NC}"
    exit 1
fi

# PostgreSQL connection string
PG_CONN="-h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER"

echo -e "${YELLOW}=== TalentCo Database Reset ===${NC}"

echo -e "${YELLOW}Removing all migrations...${NC}"
find apps -path "*/migrations/*.py" -not -name "__init__.py" -delete
find apps -path "*/migrations/*.pyc" -delete

echo -e "${YELLOW}Terminating existing database connections...${NC}"
PGPASSWORD=$POSTGRES_PASSWORD psql $PG_CONN -c "
SELECT pg_terminate_backend(pg_stat_activity.pid)
FROM pg_stat_activity
WHERE pg_stat_activity.datname IN ('$POSTGRES_DB', 'test_$POSTGRES_DB')
  AND pid <> pg_backend_pid();" 2>/dev/null || true

echo -e "${YELLOW}Dropping test database...${NC}"
PGPASSWORD=$POSTGRES_PASSWORD dropdb $PG_CONN test_$POSTGRES_DB --if-exists 2>/dev/null || true

echo -e "${YELLOW}Dropping main database...${NC}"
PGPASSWORD=$POSTGRES_PASSWORD dropdb $PG_CONN $POSTGRES_DB --if-exists

echo -e "${YELLOW}Creating main database...${NC}"
PGPASSWORD=$POSTGRES_PASSWORD createdb $PG_CONN $POSTGRES_DB

echo -e "${GREEN}Creating pgvector extension BEFORE migrations...${NC}"
PGPASSWORD=$POSTGRES_PASSWORD psql $PG_CONN -d $POSTGRES_DB -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo -e "${YELLOW}Creating test database...${NC}"
PGPASSWORD=$POSTGRES_PASSWORD createdb $PG_CONN test_$POSTGRES_DB

echo -e "${YELLOW}Creating pgvector extension in test database...${NC}"
PGPASSWORD=$POSTGRES_PASSWORD psql $PG_CONN -d test_$POSTGRES_DB -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo -e "${YELLOW}Making migrations...${NC}"
python manage.py makemigrations

echo -e "${YELLOW}Applying migrations...${NC}"
python manage.py migrate

echo -e "${GREEN}Reset complete!${NC}"
echo -e "${YELLOW}Note: You may need to recreate any initial data or superuser accounts.${NC}"

# Optional: Create superuser if DJANGO_SUPERUSER_* environment variables are set
if [ ! -z "$DJANGO_SUPERUSER_USERNAME" ] && [ ! -z "$DJANGO_SUPERUSER_EMAIL" ] && [ ! -z "$DJANGO_SUPERUSER_PASSWORD" ]; then
    echo -e "${YELLOW}Creating superuser...${NC}"
    python manage.py createsuperuser --noinput
fi

echo -e "${GREEN}Done!${NC}"