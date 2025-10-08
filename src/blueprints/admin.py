"""Admin blueprint (Task 12)

Includes user management, settings, stats, migration, auto-process and inquire admin endpoints.
Extracted from `app.py` with minimal structural changes.
"""
from __future__ import annotations

import os
from datetime import datetime
from flask import Blueprint, jsonify, request, current_app, render_template
from flask_login import login_required, current_user

from src.extensions import db
from src.models import User, SystemSetting, Recording, InquireSession

admin_bp = Blueprint('admin', __name__)


def _admin_required():
    if not current_user.is_authenticated or not current_user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    return None


@admin_bp.route('/admin', methods=['GET'])
@login_required
def admin():
    denied = _admin_required()
    if denied: return denied
    return render_template('admin.html')


@admin_bp.route('/admin/users', methods=['GET'])
@login_required
def admin_list_users():
    denied = _admin_required()
    if denied: return denied
    users = User.query.order_by(User.created_at.desc()).all()
    return jsonify([u.to_dict(include_email=True) for u in users])


@admin_bp.route('/admin/users', methods=['POST'])
@login_required
def admin_create_user():
    denied = _admin_required()
    if denied: return denied
    data = request.json or {}
    required = ['username', 'email', 'password']
    if any(k not in data or not data[k] for k in required):
        return jsonify({'error': 'username, email, and password are required'}), 400
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400
    user = User(username=data['username'], email=data['email'])
    from src.extensions import bcrypt
    user.password_hash = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    user.is_admin = data.get('is_admin', False)
    db.session.add(user); db.session.commit()
    return jsonify(user.to_dict(include_email=True)), 201


@admin_bp.route('/admin/users/<int:user_id>', methods=['PUT'])
@login_required
def admin_update_user(user_id):
    denied = _admin_required()
    if denied: return denied
    user = db.session.get(User, user_id)
    if not user: return jsonify({'error': 'User not found'}), 404
    data = request.json or {}
    if 'email' in data: user.email = data['email']
    if 'is_admin' in data: user.is_admin = data['is_admin']
    if 'output_language' in data: user.output_language = data['output_language']
    if 'transcription_language' in data: user.transcription_language = data['transcription_language']
    user.updated_at = datetime.utcnow(); db.session.commit(); return jsonify(user.to_dict(include_email=True))


@admin_bp.route('/admin/users/<int:user_id>', methods=['DELETE'])
@login_required
def admin_delete_user(user_id):
    denied = _admin_required()
    if denied: return denied
    
    # Prevent deleting self
    if user_id == current_user.id:
        return jsonify({'error': 'Cannot delete your own account'}), 400
    
    user = db.session.get(User, user_id)
    if not user: return jsonify({'error': 'User not found'}), 404
    
    # Delete user's recordings and audio files
    total_chunks = 0
    ENABLE_INQUIRE_MODE = os.environ.get('ENABLE_INQUIRE_MODE', 'false').lower() == 'true'
    if ENABLE_INQUIRE_MODE:
        from src.models import TranscriptChunk
        total_chunks = TranscriptChunk.query.filter_by(user_id=user_id).count()
        if total_chunks > 0:
            current_app.logger.info(f"Deleting {total_chunks} transcript chunks with embeddings for user {user_id}")
    
    for recording in user.recordings:
        try:
            if recording.audio_path and os.path.exists(recording.audio_path):
                os.remove(recording.audio_path)
        except Exception as e:
            current_app.logger.error(f"Error deleting audio file {recording.audio_path}: {e}")
        
        # Delete attachment files if they exist
        if recording.attachments:
            for attachment in recording.attachments:
                try:
                    if attachment.file_path and os.path.exists(attachment.file_path):
                        os.remove(attachment.file_path)
                except Exception as e:
                    current_app.logger.error(f"Error deleting attachment file {attachment.file_path}: {e}")
    
    # Delete user (cascade will handle all related data including chunks/embeddings)
    db.session.delete(user)
    db.session.commit()
    
    if ENABLE_INQUIRE_MODE and total_chunks > 0:
        current_app.logger.info(f"Successfully deleted {total_chunks} embeddings and chunks for user {user_id}")
    
    return jsonify({'success': True})


@admin_bp.route('/admin/users/<int:user_id>/toggle-admin', methods=['POST'])
@login_required
def admin_toggle_user_admin(user_id):
    denied = _admin_required()
    if denied: return denied
    user = db.session.get(User, user_id)
    if not user: return jsonify({'error': 'User not found'}), 404
    user.is_admin = not user.is_admin; db.session.commit(); return jsonify({'success': True, 'is_admin': user.is_admin})


@admin_bp.route('/admin/stats', methods=['GET'])
@login_required
def admin_stats():
    denied = _admin_required()
    if denied: return denied
    
    # Get total users
    total_users = User.query.count()
    
    # Get total recordings
    total_recordings = Recording.query.count()
    
    # Get recordings by status
    completed_recordings = Recording.query.filter_by(status='COMPLETED').count()
    processing_recordings = Recording.query.filter(Recording.status.in_(['PROCESSING', 'SUMMARIZING'])).count()
    pending_recordings = Recording.query.filter_by(status='PENDING').count()
    failed_recordings = Recording.query.filter_by(status='FAILED').count()
    
    # Get total storage used
    total_storage = db.session.query(db.func.sum(Recording.file_size)).scalar() or 0
    
    # Get top users by storage
    top_users_query = db.session.query(
        User.id,
        User.username,
        db.func.count(Recording.id).label('recordings_count'),
        db.func.sum(Recording.file_size).label('storage_used')
    ).join(Recording, User.id == Recording.user_id, isouter=True) \
     .group_by(User.id) \
     .order_by(db.func.sum(Recording.file_size).desc()) \
     .limit(5)
    
    top_users = []
    for user_id, username, recordings_count, storage_used in top_users_query:
        top_users.append({
            'id': user_id,
            'username': username,
            'recordings_count': recordings_count or 0,
            'storage_used': storage_used or 0
        })
    
    # Get total inquire sessions
    total_sessions = InquireSession.query.count()
    
    # Get total queries (chat requests)
    # This is a placeholder - you would need to track this in your database
    total_queries = 0
    
    return jsonify({
        'users': total_users,
        'recordings': total_recordings,
        'completed_recordings': completed_recordings,
        'processing_recordings': processing_recordings,
        'pending_recordings': pending_recordings,
        'failed_recordings': failed_recordings,
        'total_storage': total_storage,
        'top_users': top_users,
        'inquire_sessions': total_sessions,
        'total_queries': total_queries
    })


@admin_bp.route('/admin/settings', methods=['GET'])
@login_required
def admin_get_settings():
    denied = _admin_required()
    if denied: return denied
    settings = SystemSetting.query.all()
    return jsonify([s.to_dict() for s in settings])


@admin_bp.route('/admin/settings', methods=['POST'])
@login_required
def admin_update_settings():
    denied = _admin_required()
    if denied: return denied
    data = request.json or {}
    updated = []
    for key, value in data.items():
        SystemSetting.set_setting(key=key, value=value)
        updated.append(key)
    return jsonify({'success': True, 'updated': updated})


# --- Migration / Auto Process ---
@admin_bp.route('/api/admin/migrate_recordings', methods=['POST'])
@login_required
def migrate_recordings():
    denied = _admin_required()
    if denied: return denied
    
    try:
        from src.services.llm_service import process_recording_chunks
        from src.models import TranscriptChunk
        
        # Count recordings that need processing
        completed_recordings = Recording.query.filter_by(status='COMPLETED').all()
        recordings_needing_processing = []
        
        for recording in completed_recordings:
            if recording.transcription:  # Has transcription
                chunk_count = TranscriptChunk.query.filter_by(recording_id=recording.id).count()
                if chunk_count == 0:  # No chunks yet
                    recordings_needing_processing.append(recording)
        
        if len(recordings_needing_processing) == 0:
            return jsonify({
                'success': True,
                'message': 'All recordings are already processed for inquire mode',
                'processed': 0,
                'total': len(completed_recordings)
            })
        
        # Process in small batches to avoid timeout
        batch_size = min(5, len(recordings_needing_processing))  # Process max 5 at a time
        processed = 0
        errors = 0
        
        for i in range(min(batch_size, len(recordings_needing_processing))):
            recording = recordings_needing_processing[i]
            try:
                success = process_recording_chunks(recording.id)
                if success:
                    processed += 1
                else:
                    errors += 1
            except Exception as e:
                current_app.logger.error(f"Error processing recording {recording.id} for migration: {e}")
                errors += 1
        
        remaining = max(0, len(recordings_needing_processing) - batch_size)
        
        return jsonify({
            'success': True,
            'message': f'Processed {processed} recordings. {remaining} remaining.',
            'processed': processed,
            'errors': errors,
            'remaining': remaining,
            'total': len(recordings_needing_processing)
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in migration API: {e}")
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/admin/auto-process/status', methods=['GET'])
@login_required
def auto_process_status():
    denied = _admin_required()
    if denied: return denied
    
    try:
        from src.file_monitor import get_file_monitor_status
        status = get_file_monitor_status()
        
        # Add configuration info
        config = {
            'enabled': os.environ.get('ENABLE_AUTO_PROCESSING', 'false').lower() == 'true',
            'watch_directory': os.environ.get('AUTO_PROCESS_WATCH_DIR', '/data/auto-process'),
            'check_interval': int(os.environ.get('AUTO_PROCESS_CHECK_INTERVAL', '30')),
            'mode': os.environ.get('AUTO_PROCESS_MODE', 'admin_only'),
            'default_username': os.environ.get('AUTO_PROCESS_DEFAULT_USERNAME')
        }
        
        return jsonify({
            'status': status,
            'config': config
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting auto-process status: {e}")
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/admin/auto-process/start', methods=['POST'])
@login_required
def auto_process_start():
    denied = _admin_required()
    if denied: return denied
    
    try:
        from src.file_monitor import start_file_monitor
        start_file_monitor()
        return jsonify({'success': True, 'message': 'Auto-processing started'})
    except Exception as e:
        current_app.logger.error(f"Error starting auto-process: {e}")
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/admin/auto-process/stop', methods=['POST'])
@login_required
def auto_process_stop():
    denied = _admin_required()
    if denied: return denied
    
    try:
        from src.file_monitor import stop_file_monitor
        stop_file_monitor()
        return jsonify({'success': True, 'message': 'Auto-processing stopped'})
    except Exception as e:
        current_app.logger.error(f"Error stopping auto-process: {e}")
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/admin/auto-process/config', methods=['POST'])
@login_required
def auto_process_config():
    denied = _admin_required()
    if denied: return denied
    data = request.json or {}
    return jsonify({'success': True, 'received': data})


@admin_bp.route('/admin/inquire/process-recordings', methods=['POST'])
@login_required
def admin_inquire_process_recordings():
    denied = _admin_required()
    if denied: return denied
    return jsonify({'success': True, 'message': 'Processing recordings for inquire (placeholder)'})


@admin_bp.route('/admin/inquire/status', methods=['GET'])
@login_required
def admin_inquire_status():
    denied = _admin_required()
    if denied: return denied
    return jsonify({'status': 'idle', 'processed': 0})
