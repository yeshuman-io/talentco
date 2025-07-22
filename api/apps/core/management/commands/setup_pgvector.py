"""
Django management command to enable pgvector extension in PostgreSQL.
"""
from django.core.management.base import BaseCommand
from django.db import connection


class Command(BaseCommand):
    help = 'Enable pgvector extension in PostgreSQL database'

    def handle(self, *args, **options):
        """Enable pgvector extension if not already enabled."""
        try:
            with connection.cursor() as cursor:
                # Check if extension already exists
                cursor.execute(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');"
                )
                extension_exists = cursor.fetchone()[0]
                
                if not extension_exists:
                    # Create the extension
                    cursor.execute("CREATE EXTENSION vector;")
                    self.stdout.write(
                        self.style.SUCCESS('Successfully enabled pgvector extension')
                    )
                else:
                    self.stdout.write(
                        self.style.WARNING('pgvector extension already enabled')
                    )
                    
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Failed to enable pgvector extension: {e}')
            )
            # Don't fail the deployment if extension can't be created
            # (it might already exist or be created by a superuser)
            self.stdout.write(
                self.style.WARNING('Continuing deployment...')
            ) 