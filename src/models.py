"""
Database models for LecApp.

This module contains all SQLAlchemy database models and Flask-WTF forms
used throughout the application.
"""

from datetime import datetime
from flask_login import UserMixin
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
import secrets
import json
import re

from src.extensions import db
from src.utils import md_to_html


# --- Custom Validators ---

def password_check(form, field):
    """Custom password validator for strong passwords."""
    password = field.data
    if len(password) < 8:
        raise ValidationError('Password must be at least 8 characters long.')
    if not re.search(r'[A-Z]', password):
        raise ValidationError('Password must contain at least one uppercase letter.')
    if not re.search(r'[a-z]', password):
        raise ValidationError('Password must contain at least one lowercase letter.')
    if not re.search(r'[0-9]', password):
        raise ValidationError('Password must contain at least one number.')
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        raise ValidationError('Password must contain at least one special character.')


# --- Authentication Forms ---

class RegistrationForm(FlaskForm):
    """User registration form."""
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), password_check])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')
    
    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('That username is already taken. Please choose a different one.')
    
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('That email is already registered. Please use a different one.')


class LoginForm(FlaskForm):
    """User login form."""
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')


# --- Database Models ---

class User(db.Model, UserMixin):
    """User model for authentication and preferences."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    recordings = db.relationship('Recording', backref='owner', lazy=True)
    transcription_language = db.Column(db.String(10), nullable=True)
    output_language = db.Column(db.String(50), nullable=True)
    ui_language = db.Column(db.String(10), nullable=True, default='en')
    summary_prompt = db.Column(db.Text, nullable=True)
    extract_events = db.Column(db.Boolean, default=False)
    name = db.Column(db.String(100), nullable=True)
    job_title = db.Column(db.String(100), nullable=True)
    company = db.Column(db.String(100), nullable=True)
    diarize = db.Column(db.Boolean, default=False)
    
    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"


class Speaker(db.Model):
    """Speaker model for managing identified speakers."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_used = db.Column(db.DateTime, default=datetime.utcnow)
    use_count = db.Column(db.Integer, default=1)
    
    user = db.relationship('User', backref=db.backref('speakers', lazy=True, cascade='all, delete-orphan'))
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at,
            'last_used': self.last_used,
            'use_count': self.use_count
        }


class SystemSetting(db.Model):
    """System-wide configuration settings."""
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.Text, nullable=True)
    description = db.Column(db.Text, nullable=True)
    setting_type = db.Column(db.String(50), nullable=False, default='string')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'key': self.key,
            'value': self.value,
            'description': self.description,
            'setting_type': self.setting_type,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @staticmethod
    def get_setting(key, default_value=None):
        """Get a system setting value by key, with optional default."""
        setting = SystemSetting.query.filter_by(key=key).first()
        if setting:
            # Convert value based on type
            if setting.setting_type == 'integer':
                try:
                    return int(setting.value) if setting.value is not None else default_value
                except (ValueError, TypeError):
                    return default_value
            elif setting.setting_type == 'boolean':
                return setting.value.lower() in ('true', '1', 'yes') if setting.value else default_value
            elif setting.setting_type == 'float':
                try:
                    return float(setting.value) if setting.value is not None else default_value
                except (ValueError, TypeError):
                    return default_value
            else:  # string
                return setting.value if setting.value is not None else default_value
        return default_value
    
    @staticmethod
    def set_setting(key, value, description=None, setting_type='string'):
        """Set a system setting value."""
        setting = SystemSetting.query.filter_by(key=key).first()
        if setting:
            setting.value = str(value) if value is not None else None
            setting.updated_at = datetime.utcnow()
            if description:
                setting.description = description
            if setting_type:
                setting.setting_type = setting_type
        else:
            setting = SystemSetting(
                key=key,
                value=str(value) if value is not None else None,
                description=description,
                setting_type=setting_type
            )
            db.session.add(setting)
        db.session.commit()
        return setting


class RecordingTag(db.Model):
    """Many-to-many relationship table for recordings and tags."""
    __tablename__ = 'recording_tags'
    recording_id = db.Column(db.Integer, db.ForeignKey('recording.id'), primary_key=True)
    tag_id = db.Column(db.Integer, db.ForeignKey('tag.id'), primary_key=True)
    added_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=True)
    order = db.Column(db.Integer, nullable=False, default=0)
    
    recording = db.relationship('Recording', back_populates='tag_associations')
    tag = db.relationship('Tag', back_populates='recording_associations')


class Tag(db.Model):
    """User-defined tags for organizing recordings."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    color = db.Column(db.String(7), default='#3B82F6')
    
    # Custom settings for this tag
    custom_prompt = db.Column(db.Text, nullable=True)
    default_language = db.Column(db.String(10), nullable=True)
    default_min_speakers = db.Column(db.Integer, nullable=True)
    default_max_speakers = db.Column(db.Integer, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('tags', lazy=True, cascade='all, delete-orphan'))
    recording_associations = db.relationship('RecordingTag', back_populates='tag', cascade='all, delete-orphan')
    
    __table_args__ = (db.UniqueConstraint('name', 'user_id', name='_user_tag_uc'),)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'color': self.color,
            'custom_prompt': self.custom_prompt,
            'default_language': self.default_language,
            'default_min_speakers': self.default_min_speakers,
            'default_max_speakers': self.default_max_speakers,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'recording_count': len(self.recording_associations)
        }


class Event(db.Model):
    """Calendar events extracted from recordings."""
    id = db.Column(db.Integer, primary_key=True)
    recording_id = db.Column(db.Integer, db.ForeignKey('recording.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    start_datetime = db.Column(db.DateTime, nullable=False)
    end_datetime = db.Column(db.DateTime, nullable=True)
    location = db.Column(db.String(500), nullable=True)
    attendees = db.Column(db.Text, nullable=True)
    reminder_minutes = db.Column(db.Integer, nullable=True, default=15)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    recording = db.relationship('Recording', backref=db.backref('events', lazy=True, cascade='all, delete-orphan'))

    def to_dict(self):
        return {
            'id': self.id,
            'recording_id': self.recording_id,
            'title': self.title,
            'description': self.description,
            'start_datetime': self.start_datetime.isoformat() if self.start_datetime else None,
            'end_datetime': self.end_datetime.isoformat() if self.end_datetime else None,
            'location': self.location,
            'attendees': json.loads(self.attendees) if self.attendees else [],
            'reminder_minutes': self.reminder_minutes,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Share(db.Model):
    """Public sharing links for recordings."""
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(32), unique=True, nullable=False, default=lambda: secrets.token_urlsafe(16))
    recording_id = db.Column(db.Integer, db.ForeignKey('recording.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    share_summary = db.Column(db.Boolean, default=True)
    share_notes = db.Column(db.Boolean, default=True)

    user = db.relationship('User', backref=db.backref('shares', lazy=True, cascade='all, delete-orphan'))
    recording = db.relationship('Recording', backref=db.backref('shares', lazy=True, cascade='all, delete-orphan'))

    def to_dict(self):
        # Import here to avoid circular dependency
        from babel.dates import format_datetime
        import pytz
        import os
        
        # Get timezone from .env, default to UTC
        user_tz_name = os.environ.get('TIMEZONE', 'UTC')
        try:
            user_tz = pytz.timezone(user_tz_name)
        except pytz.UnknownTimeZoneError:
            user_tz = pytz.utc
        
        # Format datetime
        if self.created_at.tzinfo is None:
            dt = pytz.utc.localize(self.created_at)
        else:
            dt = self.created_at
        local_dt = dt.astimezone(user_tz)
        formatted_date = format_datetime(local_dt, format='medium', locale='en_US')
        
        return {
            'id': self.id,
            'public_id': self.public_id,
            'recording_id': self.recording_id,
            'created_at': formatted_date,
            'share_summary': self.share_summary,
            'share_notes': self.share_notes,
            'recording_title': self.recording.title if self.recording else "N/A"
        }


class Recording(db.Model):
    """Main recording model."""
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=True)
    participants = db.Column(db.String(500))
    notes = db.Column(db.Text)
    transcription = db.Column(db.Text, nullable=True)
    summary = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(50), default='PENDING')
    audio_path = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    meeting_date = db.Column(db.Date, nullable=True)
    file_size = db.Column(db.Integer)
    original_filename = db.Column(db.String(500), nullable=True)
    is_inbox = db.Column(db.Boolean, default=True)
    is_highlighted = db.Column(db.Boolean, default=False)
    mime_type = db.Column(db.String(100), nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    processing_time_seconds = db.Column(db.Integer, nullable=True)
    processing_source = db.Column(db.String(50), default='upload')
    error_message = db.Column(db.Text, nullable=True)
    
    tag_associations = db.relationship('RecordingTag', back_populates='recording', cascade='all, delete-orphan', order_by='RecordingTag.order')
    
    @property
    def tags(self):
        """Get tags ordered by the order they were added to this recording."""
        return [assoc.tag for assoc in sorted(self.tag_associations, key=lambda x: x.order)]

    def to_dict(self):
        # Import here to avoid circular dependency
        from babel.dates import format_datetime
        import pytz
        import os
        
        # Get timezone from .env, default to UTC
        user_tz_name = os.environ.get('TIMEZONE', 'UTC')
        try:
            user_tz = pytz.timezone(user_tz_name)
        except pytz.UnknownTimeZoneError:
            user_tz = pytz.utc
        
        def format_dt(dt):
            if dt is None:
                return None
            if dt.tzinfo is None:
                dt = pytz.utc.localize(dt)
            local_dt = dt.astimezone(user_tz)
            return format_datetime(local_dt, format='medium', locale='en_US')
        
        return {
            'id': self.id,
            'title': self.title,
            'participants': self.participants,
            'notes': self.notes,
            'notes_html': md_to_html(self.notes) if self.notes else "",
            'transcription': self.transcription,
            'summary': self.summary,
            'summary_html': md_to_html(self.summary) if self.summary else "",
            'status': self.status,
            'created_at': format_dt(self.created_at),
            'completed_at': format_dt(self.completed_at),
            'processing_time_seconds': self.processing_time_seconds,
            'meeting_date': f"{self.meeting_date.isoformat()}T00:00:00" if self.meeting_date else None,
            'file_size': self.file_size,
            'original_filename': self.original_filename,
            'user_id': self.user_id,
            'is_inbox': self.is_inbox,
            'is_highlighted': self.is_highlighted,
            'mime_type': self.mime_type,
            'tags': [tag.to_dict() for tag in self.tags] if self.tags else [],
            'events': [event.to_dict() for event in self.events] if self.events else []
        }


class TranscriptChunk(db.Model):
    """Stores chunked transcription segments for efficient retrieval and embedding."""
    id = db.Column(db.Integer, primary_key=True)
    recording_id = db.Column(db.Integer, db.ForeignKey('recording.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    chunk_index = db.Column(db.Integer, nullable=False)
    content = db.Column(db.Text, nullable=False)
    start_time = db.Column(db.Float, nullable=True)
    end_time = db.Column(db.Float, nullable=True)
    speaker_name = db.Column(db.String(100), nullable=True)
    embedding = db.Column(db.LargeBinary, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    recording = db.relationship('Recording', backref=db.backref('chunks', lazy=True, cascade='all, delete-orphan'))
    user = db.relationship('User', backref=db.backref('transcript_chunks', lazy=True, cascade='all, delete-orphan'))
    
    def to_dict(self):
        return {
            'id': self.id,
            'recording_id': self.recording_id,
            'chunk_index': self.chunk_index,
            'content': self.content,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'speaker_name': self.speaker_name,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class TranscriptTemplate(db.Model):
    """Stores user-defined templates for transcript formatting."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    template = db.Column(db.Text, nullable=False)
    description = db.Column(db.String(500), nullable=True)
    is_default = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('transcript_templates', lazy=True, cascade='all, delete-orphan'))

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'template': self.template,
            'description': self.description,
            'is_default': self.is_default,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class Attachment(db.Model):
    """Stores supporting documents attached to recordings."""
    id = db.Column(db.Integer, primary_key=True)
    recording_id = db.Column(db.Integer, db.ForeignKey('recording.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    original_filename = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    file_type = db.Column(db.String(50), nullable=True)
    mime_type = db.Column(db.String(100), nullable=True)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    recording = db.relationship('Recording', backref=db.backref('attachments', lazy=True, cascade='all, delete-orphan'))
    user = db.relationship('User', backref=db.backref('attachments', lazy=True, cascade='all, delete-orphan'))
    
    def to_dict(self):
        return {
            'id': self.id,
            'recording_id': self.recording_id,
            'original_filename': self.original_filename,
            'file_size': self.file_size,
            'file_type': self.file_type,
            'mime_type': self.mime_type,
            'uploaded_at': self.uploaded_at.isoformat() if self.uploaded_at else None
        }


class InquireSession(db.Model):
    """Tracks inquire mode sessions and their filtering criteria."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_name = db.Column(db.String(200), nullable=True)
    
    # Filter criteria (JSON stored as text)
    filter_tags = db.Column(db.Text, nullable=True)
    filter_speakers = db.Column(db.Text, nullable=True)
    filter_date_from = db.Column(db.Date, nullable=True)
    filter_date_to = db.Column(db.Date, nullable=True)
    filter_recording_ids = db.Column(db.Text, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_used = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('inquire_sessions', lazy=True, cascade='all, delete-orphan'))
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_name': self.session_name,
            'filter_tags': json.loads(self.filter_tags) if self.filter_tags else [],
            'filter_speakers': json.loads(self.filter_speakers) if self.filter_speakers else [],
            'filter_date_from': self.filter_date_from.isoformat() if self.filter_date_from else None,
            'filter_date_to': self.filter_date_to.isoformat() if self.filter_date_to else None,
            'filter_recording_ids': json.loads(self.filter_recording_ids) if self.filter_recording_ids else [],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_used': self.last_used.isoformat() if self.last_used else None
        }
