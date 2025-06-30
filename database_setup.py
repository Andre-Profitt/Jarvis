"""
JARVIS Database Migration Setup
Initializes Alembic for database version control
"""

import os
import sys
from pathlib import Path
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    Float,
    Boolean,
    JSON,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

Base = declarative_base()


# Define all database models
class ConsciousnessState(Base):
    __tablename__ = "consciousness_states"

    id = Column(Integer, primary_key=True)
    awareness_level = Column(Float, default=0.5)
    active_thoughts = Column(JSON)
    memory_access = Column(Boolean, default=True)
    learning_rate = Column(Float, default=0.01)
    timestamp = Column(DateTime, default=datetime.utcnow)
    context = Column(JSON)


class ScheduledJob(Base):
    __tablename__ = "scheduled_jobs"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True)
    schedule = Column(String(100))  # Cron expression
    task = Column(String(255))
    params = Column(JSON)
    enabled = Column(Boolean, default=True)
    last_run = Column(DateTime)
    next_run = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)


class ServiceHealth(Base):
    __tablename__ = "service_health"

    id = Column(Integer, primary_key=True)
    service_name = Column(String(100))
    status = Column(String(50))  # healthy, degraded, down
    latency = Column(Float)
    error_rate = Column(Float)
    last_check = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)


class KnowledgeEntry(Base):
    __tablename__ = "knowledge_entries"

    id = Column(Integer, primary_key=True)
    content = Column(Text)
    embeddings = Column(JSON)  # Store vector embeddings
    metadata = Column(JSON)
    project_id = Column(String(100))
    timestamp = Column(DateTime, default=datetime.utcnow)


class FeatureFlag(Base):
    __tablename__ = "feature_flags"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True)
    enabled = Column(Boolean, default=False)
    description = Column(Text)
    conditions = Column(JSON)  # Complex targeting conditions
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AnomalyDetection(Base):
    __tablename__ = "anomaly_detections"

    id = Column(Integer, primary_key=True)
    service = Column(String(100))
    metric_name = Column(String(100))
    value = Column(Float)
    threshold = Column(Float)
    severity = Column(String(50))  # low, medium, high, critical
    detected_at = Column(DateTime, default=datetime.utcnow)
    resolved = Column(Boolean, default=False)
    metadata = Column(JSON)


class CircuitBreakerState(Base):
    __tablename__ = "circuit_breaker_states"

    id = Column(Integer, primary_key=True)
    service = Column(String(100), unique=True)
    state = Column(String(50))  # closed, open, half_open
    failure_count = Column(Integer, default=0)
    last_failure = Column(DateTime)
    last_success = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SystemMetric(Base):
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True)
    metric_type = Column(String(100))
    value = Column(Float)
    unit = Column(String(50))
    tags = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)


def setup_alembic():
    """Initialize Alembic for database migrations"""

    # Create alembic directory structure
    alembic_dir = Path("alembic")
    if not alembic_dir.exists():
        os.system("alembic init alembic")

    # Configure alembic.ini
    alembic_ini_content = """# A generic, single database configuration.

[alembic]
# path to migration scripts
script_location = alembic

# template used to generate migration file names; The default value is %%(rev)s_%%(slug)s
# Uncomment the line below if you want the files to be prepended with date and time
file_template = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d-%%(rev)s_%%(slug)s

# sys.path path, will be prepended to sys.path if present.
# defaults to the current working directory.
prepend_sys_path = .

# timezone to use when rendering the date within the migration file
# as well as the filename.
# If specified, requires the python-dateutil library
# timezone =

# max length of characters to apply to the
# "slug" field
# truncate_slug_length = 40

# set to 'true' to run the environment during
# the 'revision' command, regardless of autogenerate
# revision_environment = false

# set to 'true' to allow .pyc and .pyo files without
# a source .py file to be detected as revisions in the
# versions/ directory
# sourceless = false

# version location specification; This defaults
# to alembic/versions.  When using multiple version
# directories, initial revisions must be specified with --version-path.
# The path separator used here should be the separator specified by "version_path_separator" below.
# version_locations = %(here)s/bar:%(here)s/bat:alembic/versions

# version path separator; As mentioned above, this is the character used to split
# version_locations. The default within new alembic.ini files is "os", which uses os.pathsep.
# If this key is omitted entirely, it falls back to the legacy behavior of splitting on spaces and/or commas.
# Valid values for version_path_separator are:
#
# version_path_separator = :
# version_path_separator = ;
# version_path_separator = space
version_path_separator = os  # Use os.pathsep.
# the output encoding used when revision files
# are written from script.py.mako
# output_encoding = utf-8

sqlalchemy.url = sqlite:///jarvis.db


[post_write_hooks]
# post_write_hooks defines scripts or Python functions that are run
# on newly generated revision scripts.  See the documentation for further
# detail and examples

# format using "black" - use the console_scripts runner, against the "black" entrypoint
# hooks = black
# black.type = console_scripts
# black.entrypoint = black
# black.options = -l 79 REVISION_SCRIPT_FILENAME

# Logging configuration
[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s:
datefmt = %H:%M:%S
"""

    with open("alembic.ini", "w") as f:
        f.write(alembic_ini_content)

    # Update env.py in alembic
    env_py_content = '''"""Alembic environment script"""
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import your models
from database_setup import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

def get_database_url():
    """Get database URL from environment or use default"""
    return os.getenv('DATABASE_URL', 'sqlite:///jarvis.db')

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    configuration = config.get_section(config.config_ini_section)
    configuration['sqlalchemy.url'] = get_database_url()
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'''

    # Write env.py
    env_path = Path("alembic/env.py")
    if env_path.parent.exists():
        with open(env_path, "w") as f:
            f.write(env_py_content)

    print("✅ Alembic setup complete!")
    print("📝 Next steps:")
    print("1. Run: alembic revision --autogenerate -m 'Initial migration'")
    print("2. Run: alembic upgrade head")


if __name__ == "__main__":
    setup_alembic()
