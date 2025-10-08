"""
Flask extensions initialization.

This module initializes all Flask extensions used in the application.
Extensions are created here without app binding, allowing for deferred
initialization in the application factory pattern.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect

# Initialize extensions (without app binding - deferred initialization)
db = SQLAlchemy()

login_manager = LoginManager()

bcrypt = Bcrypt()

# Rate limiting setup
# TEMPORARILY INCREASED FOR TESTING - REVERT FOR PRODUCTION!
limiter = Limiter(
    get_remote_address,
    default_limits=["5000 per day", "1000 per hour"]  # Increased from 200/day, 50/hour for testing
)

csrf = CSRFProtect()