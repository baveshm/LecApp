"""Authentication blueprint extracting auth-related routes from app.py.

Includes: register, login, logout, account management, change password, user preferences.
"""
import os
from urllib.parse import urlparse, urljoin
from datetime import datetime

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_user, logout_user, login_required, current_user

from src.extensions import db, bcrypt, limiter, csrf
from src.models import User, SystemSetting, RegistrationForm, LoginForm

auth_bp = Blueprint('auth', __name__)


def _is_safe_url(target):
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc


@auth_bp.route('/register', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def register():
    allow_registration = os.environ.get('ALLOW_REGISTRATION', 'true').lower() == 'true'
    if not allow_registration:
        flash('Registration is currently disabled. Please contact the administrator.', 'danger')
        return redirect(url_for('auth.login'))
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('auth.login'))
    return render_template('register.html', title='Register', form=form)


@auth_bp.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            if not _is_safe_url(next_page):
                return redirect(url_for('index'))
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Login unsuccessful. Please check email and password.', 'danger')
    return render_template('login.html', title='Login', form=form)


@auth_bp.route('/logout')
@csrf.exempt
def logout():
    logout_user()
    return redirect(url_for('auth.login'))


@auth_bp.route('/account', methods=['GET', 'POST'])
@login_required
def account():
    # Get default summary prompt for template
    default_summary_prompt_text = SystemSetting.get_setting('admin_default_summary_prompt', 
        'Create a comprehensive summary of the transcription. Include key points, decisions made, and action items.')
    
    if request.method == 'POST':
        # Update profile fields
        current_user.name = request.form.get('user_name') or current_user.name
        current_user.job_title = request.form.get('user_job_title') or current_user.job_title
        current_user.company = request.form.get('user_company') or current_user.company
        
        # Update language preferences
        current_user.ui_language = request.form.get('ui_language') or current_user.ui_language
        current_user.transcription_language = request.form.get('transcription_language') or current_user.transcription_language
        current_user.output_language = request.form.get('output_language') or current_user.output_language
        
        # Update prompt options
        current_user.summary_prompt = request.form.get('summary_prompt') or current_user.summary_prompt
        current_user.extract_events = 'extract_events' in request.form
        
        db.session.commit()
        flash('Account updated successfully.', 'success')
        return redirect(url_for('auth.account'))
    
    return render_template('account.html', title='Account', 
                         default_summary_prompt_text=default_summary_prompt_text)


@auth_bp.route('/change_password', methods=['POST'])
@login_required
def change_password():
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    if not all([current_password, new_password, confirm_password]):
        flash('All password fields are required.', 'danger')
        return redirect(url_for('auth.account'))
    if new_password != confirm_password:
        flash('New passwords do not match.', 'danger')
        return redirect(url_for('auth.account'))
    if not bcrypt.check_password_hash(current_user.password, current_password):
        flash('Current password is incorrect.', 'danger')
        return redirect(url_for('auth.account'))
    hashed = bcrypt.generate_password_hash(new_password).decode('utf-8')
    current_user.password = hashed
    db.session.commit()
    flash('Password updated successfully.', 'success')
    return redirect(url_for('auth.account'))


@auth_bp.route('/api/user/preferences', methods=['POST'])
@login_required
def save_preferences():
    data = request.get_json(silent=True) or {}
    updated = False
    for field in ['ui_language', 'output_language', 'transcription_language', 'summary_prompt', 'extract_events', 'diarize']:
        if field in data:
            setattr(current_user, field, data[field])
            updated = True
    if updated:
        db.session.commit()
    return jsonify({'success': True})
