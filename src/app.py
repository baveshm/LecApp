"""LecApp Application Factory (Task 13)

Minimal factory implementation - all feature routes are in blueprints.
Completed: blueprints for auth, recordings, attachments, inquire, tags_shares, admin.
"""
from __future__ import annotations

import logging
import mimetypes
import os
import sys
from datetime import datetime

import pytz
from babel.dates import format_datetime
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template
from flask_login import current_user, login_required
from werkzeug.middleware.proxy_fix import ProxyFix

from src.extensions import bcrypt, csrf, db, limiter, login_manager
from src.models import SystemSetting, User
from src.services.llm_service import (
    EMBEDDINGS_AVAILABLE,
    TEXT_MODEL_BASE_URL,
    TEXT_MODEL_NAME,
)
from src.services.transcription_service import (
    ASR_BASE_URL,
    USE_ASR_ENDPOINT,
    transcription_base_url,
)

# Load .env from project root (parent directory of src/)
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(_env_path)
ENABLE_INQUIRE_MODE = os.environ.get("ENABLE_INQUIRE_MODE", "false").lower() == "true"

# Register common audio MIME types
for _mt, _ext in [
    ("audio/mp4", ".m4a"),
    ("audio/aac", ".aac"),
    ("audio/x-m4a", ".m4a"),
    ("audio/webm", ".webm"),
    ("audio/flac", ".flac"),
    ("audio/ogg", ".ogg"),
]:
    mimetypes.add_type(_mt, _ext)


def _configure_logging() -> None:
    """Configure application logging."""
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(level)
        root.addHandler(handler)
    wlog = logging.getLogger("werkzeug")
    if not wlog.handlers:
        wlog.setLevel(level)
        wlog.addHandler(handler)


def get_version() -> str:
    """Get application version from VERSION file or git."""
    try:
        with open("VERSION", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        pass
    try:
        import subprocess
        return subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _log_transcription_config(app: Flask) -> None:
    """Log transcription service configuration."""
    if USE_ASR_ENDPOINT:
        if not ASR_BASE_URL:
            app.logger.error("ASR endpoint enabled but ASR_BASE_URL not set")
        else:
            app.logger.info(f"Using ASR endpoint: {ASR_BASE_URL}")
    else:
        if not transcription_base_url:
            app.logger.warning("Whisper API base URL not configured")
        else:
            app.logger.info(f"Using Whisper API at: {transcription_base_url}")


def _seed_settings(app: Flask) -> None:
    """Initialize default system settings."""
    try:
        if not SystemSetting.query.filter_by(key="transcript_length_limit").first():
            SystemSetting.set_setting("transcript_length_limit", "30000", "Maximum transcript chars for LLM (-1 unlimited)", "integer")
        if not SystemSetting.query.filter_by(key="max_file_size_mb").first():
            SystemSetting.set_setting("max_file_size_mb", "250", "Maximum upload file size (MB)", "integer")
    except Exception as e:  # pragma: no cover
        app.logger.error(f"Setting seed error: {e}")


def create_app() -> Flask:
    """Application factory - creates and configures the Flask app."""
    _configure_logging()
    
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    
    # Get project root (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Resolve paths from environment - convert relative paths to absolute
    db_uri = os.environ.get("SQLALCHEMY_DATABASE_URI", "sqlite:////data/instance/transcriptions.db")
    upload_folder = os.environ.get("UPLOAD_FOLDER", "/data/uploads")
    
    # If paths are relative, make them absolute from project root
    if not db_uri.startswith('sqlite:///'):
        # Non-SQLite URI, leave as is
        pass
    else:
        # Extract path from sqlite:///path
        db_path = db_uri.replace('sqlite:///', '')
        if not os.path.isabs(db_path):
            db_path = os.path.normpath(os.path.join(project_root, db_path))
            db_uri = f'sqlite:///{db_path}'
    
    if not os.path.isabs(upload_folder):
        upload_folder = os.path.normpath(os.path.join(project_root, upload_folder))
    
    # Configuration
    app.config.update(
        SQLALCHEMY_DATABASE_URI=db_uri,
        UPLOAD_FOLDER=upload_folder,
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-secret-key-change-me"),
        SESSION_COOKIE_SECURE=False,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    bcrypt.init_app(app)
    limiter.init_app(app)
    csrf.init_app(app)

    # Template context processors and filters
    @app.context_processor
    def inject_now():  # pragma: no cover
        return {"now": datetime.now()}

    @app.template_filter("localdatetime")
    def local_datetime_filter(dt):  # pragma: no cover
        if dt is None:
            return ""
        tz_name = os.environ.get("TIMEZONE", "UTC")
        try:
            tz = pytz.timezone(tz_name)
        except pytz.UnknownTimeZoneError:
            tz = pytz.utc
            app.logger.warning(f"Invalid TIMEZONE {tz_name}; using UTC")
        if dt.tzinfo is None:
            import pytz as _p
            dt = _p.utc.localize(dt)
        return format_datetime(dt.astimezone(tz), "medium", locale="en_US")

    @login_manager.user_loader
    def load_user(user_id):  # pragma: no cover
        return db.session.get(User, int(user_id))

    # Register blueprints
    from src.blueprints.auth import auth_bp
    from src.blueprints.recordings import recordings_bp
    from src.blueprints.attachments import attachments_bp
    from src.blueprints.inquire import inquire_bp
    from src.blueprints.tags_shares import tags_shares_bp
    from src.blueprints.admin import admin_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(recordings_bp)
    app.register_blueprint(attachments_bp)
    app.register_blueprint(inquire_bp)
    app.register_blueprint(tags_shares_bp)
    app.register_blueprint(admin_bp)

    # Root routes (not in blueprints)
    @app.route("/")
    @login_required
    def index():  # pragma: no cover
        lang = current_user.ui_language if current_user.is_authenticated and current_user.ui_language else "en"
        return render_template("index.html", use_asr_endpoint=USE_ASR_ENDPOINT, inquire_mode_enabled=ENABLE_INQUIRE_MODE, user_language=lang)

    @app.route("/api/config")
    def get_config():  # pragma: no cover
        try:
            max_mb = SystemSetting.get_setting("max_file_size_mb", 250)
            return jsonify({
                "max_file_size_mb": max_mb,
                "recording_disclaimer": SystemSetting.get_setting("recording_disclaimer", ""),
                "use_asr_endpoint": USE_ASR_ENDPOINT,
            })
        except Exception as e:  # pragma: no cover
            app.logger.error(f"Config fetch failed: {e}")
            return jsonify({"error": "config unavailable"}), 500

    # Database initialization
    with app.app_context():
        db.create_all()
        _seed_settings(app)
        max_mb = SystemSetting.get_setting("max_file_size_mb", 250)
        app.config["MAX_CONTENT_LENGTH"] = max_mb * 1024 * 1024

    # Log startup information
    version = get_version()
    app.logger.info(f"=== LecApp {version} Initialized ===")
    app.logger.info(f"LLM endpoint: {TEXT_MODEL_BASE_URL} model: {TEXT_MODEL_NAME}")
    if ENABLE_INQUIRE_MODE:
        if EMBEDDINGS_AVAILABLE:
            app.logger.info("Inquire Mode: semantic search enabled")
        else:
            app.logger.warning("Inquire Mode: embeddings missing; basic search only")
    else:
        app.logger.info("Inquire Mode: disabled")
    _log_transcription_config(app)
    
    return app


# Create singleton app instance (for WSGI servers and imports)
app = create_app()

if __name__ == "__main__":  # pragma: no cover
    app.run(host="0.0.0.0", port=8899)
