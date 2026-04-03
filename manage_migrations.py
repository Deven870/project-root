#!/usr/bin/env python3
"""
Database Migration Management
==============================
Handle database schema migrations and version management.

Usage:
    python manage_migrations.py init       # Initialize Alembic
    python manage_migrations.py create    # Create new migration
    python manage_migrations.py upgrade   # Apply migrations
    python manage_migrations.py downgrade # Rollback migration
"""

import sys
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from system_logger import setup_logging, get_logger

logger = get_logger(__name__)
setup_logging()


def init_migrations():
    """Initialize Alembic for the project"""
    import shutil
    from alembic.config import Config
    from alembic.command import init
    
    alembic_dir = "alembic"
    if Path(alembic_dir).exists():
        logger.warning(f"{alembic_dir} already exists")
        return
    
    try:
        # Initialize alembic directory structure
        config = Config()
        config.set_main_option("script_location", alembic_dir)
        config.set_main_option("sqlalchemy.url", "")
        
        init(config, alembic_dir, autogenerate=True)
        logger.info(f"✓ Alembic initialized in {alembic_dir}/")
        
        # Update configuration
        env_py_path = Path(alembic_dir) / "env.py"
        if env_py_path.exists():
            logger.info(f"✓ Migration environment created at {env_py_path}")
            
    except Exception as e:
        logger.error(f"✗ Failed to initialize migrations: {e}")


def create_migration(message: str):
    """Create a new migration"""
    from alembic.config import Config
    from alembic.command import revision
    
    if not Path("alembic").exists():
        logger.error("Alembic not initialized. Run 'python manage_migrations.py init' first.")
        return
    
    try:
        config = Config("alembic.ini")
        revision(config, autogenerate=True, message=message)
        logger.info(f"✓ Migration created: {message}")
    except Exception as e:
        logger.error(f"✗ Failed to create migration: {e}")


def upgrade_database():
    """Upgrade database to latest migration"""
    from alembic.config import Config
    from alembic.command import upgrade
    from database import DATABASE_URL
    
    if not Path("alembic").exists():
        logger.error("Alembic not initialized. Run 'python manage_migrations.py init' first.")
        return
    
    try:
        config = Config("alembic.ini")
        config.set_main_option("sqlalchemy.url", DATABASE_URL)
        
        upgrade(config, "head")
        logger.info("✓ Database upgraded to latest migration")
    except Exception as e:
        logger.error(f"✗ Failed to upgrade database: {e}")


def downgrade_database(revision: str = "-1"):
    """Downgrade database to previous migration"""
    from alembic.config import Config
    from alembic.command import downgrade
    from database import DATABASE_URL
    
    if not Path("alembic").exists():
        logger.error("Alembic not initialized. Run 'python manage_migrations.py init' first.")
        return
    
    try:
        config = Config("alembic.ini")
        config.set_main_option("sqlalchemy.url", DATABASE_URL)
        
        downgrade(config, revision)
        logger.info(f"✓ Database downgraded to {revision}")
    except Exception as e:
        logger.error(f"✗ Failed to downgrade database: {e}")


def reset_database():
    """Drop all tables and recreate - FOR DEVELOPMENT ONLY"""
    import os
    if os.getenv("ENVIRONMENT") != "development":
        logger.error("Cannot reset database in non-development environment")
        return
    
    from database import drop_all_tables, init_db
    
    try:
        drop_all_tables()
        init_db()
        logger.info("✓ Database reset complete")
    except Exception as e:
        logger.error(f"✗ Failed to reset database: {e}")


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database Migration Manager")
    parser.add_argument(
        "command",
        choices=["init", "create", "upgrade", "downgrade", "reset"],
        help="Migration command"
    )
    parser.add_argument(
        "--message", "-m",
        help="Migration message (for 'create' command)"
    )
    parser.add_argument(
        "--revision", "-r",
        default="-1",
        help="Revision to downgrade to"
    )
    
    args = parser.parse_args()
    
    if args.command == "init":
        init_migrations()
    elif args.command == "create":
        if not args.message:
            logger.error("--message required for 'create' command")
            sys.exit(1)
        create_migration(args.message)
    elif args.command == "upgrade":
        upgrade_database()
    elif args.command == "downgrade":
        downgrade_database(args.revision)
    elif args.command == "reset":
        reset_database()


if __name__ == "__main__":
    main()
