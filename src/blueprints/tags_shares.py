"""Tags, Shares, Transcript Templates, Preferences, System Info (Task 12).

Extracted from monolithic `app.py` keeping identical response formats.
"""
from __future__ import annotations

import os
from datetime import datetime
from flask import Blueprint, jsonify, request, url_for, render_template, make_response, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from src.extensions import db
from src.models import (
    Tag, RecordingTag, Recording, Share, TranscriptTemplate, SystemSetting, User
)
from src.utils import md_to_html

tags_shares_bp = Blueprint('tags_shares', __name__)


# --- Share Routes ---
@tags_shares_bp.route('/share/<string:public_id>', methods=['GET'])
def view_shared_recording(public_id):
    share = Share.query.filter_by(public_id=public_id).first_or_404()
    recording = share.recording
    recording_data = {
        'id': recording.id,
        'public_id': share.public_id,
        'title': recording.title,
        'participants': recording.participants,
        'transcription': recording.transcription,
        'summary': md_to_html(recording.summary) if share.share_summary else None,
        'notes': md_to_html(recording.notes) if share.share_notes else None,
        'meeting_date': f"{recording.meeting_date.isoformat()}T00:00:00" if recording.meeting_date else None,
        'mime_type': recording.mime_type
    }
    return render_template('share.html', recording=recording_data)

@tags_shares_bp.route('/api/recording/<int:recording_id>/share', methods=['GET'])
@login_required
def get_existing_share(recording_id):
    recording = db.session.get(Recording, recording_id)
    if not recording or recording.user_id != current_user.id:
        return jsonify({'error': 'Recording not found or you do not have permission to view it.'}), 404
    existing = Share.query.filter_by(recording_id=recording.id, user_id=current_user.id).order_by(Share.created_at.desc()).first()
    if existing:
        share_url = url_for('tags_shares.view_shared_recording', public_id=existing.public_id, _external=True)
        return jsonify({'success': True, 'exists': True, 'share_url': share_url, 'share': existing.to_dict()}), 200
    return jsonify({'success': True, 'exists': False}), 200

@tags_shares_bp.route('/api/recording/<int:recording_id>/share', methods=['POST'])
@login_required
def create_share(recording_id):
    if not request.is_secure:
        return jsonify({'error': 'Sharing is only available over a secure (HTTPS) connection.'}), 403
    recording = db.session.get(Recording, recording_id)
    if not recording or recording.user_id != current_user.id:
        return jsonify({'error': 'Recording not found or you do not have permission to share it.'}), 404
    data = request.json or {}
    share_summary = data.get('share_summary', True)
    share_notes = data.get('share_notes', True)
    force_new = data.get('force_new', False)
    existing = Share.query.filter_by(recording_id=recording.id, user_id=current_user.id).order_by(Share.created_at.desc()).first()
    if existing and not force_new:
        if existing.share_summary != share_summary or existing.share_notes != share_notes:
            existing.share_summary = share_summary
            existing.share_notes = share_notes
            db.session.commit()
        share_url = url_for('tags_shares.view_shared_recording', public_id=existing.public_id, _external=True)
        return jsonify({'success': True, 'share_url': share_url, 'share': existing.to_dict(), 'existing': True, 'message': 'Using existing share link for this recording'}), 200
    share = Share(recording_id=recording.id, user_id=current_user.id, share_summary=share_summary, share_notes=share_notes)
    db.session.add(share); db.session.commit()
    share_url = url_for('tags_shares.view_shared_recording', public_id=share.public_id, _external=True)
    return jsonify({'success': True, 'share_url': share_url, 'share': share.to_dict(), 'existing': False}), 201

@tags_shares_bp.route('/api/shares', methods=['GET'])
@login_required
def get_shares():
    shares = Share.query.filter_by(user_id=current_user.id).order_by(Share.created_at.desc()).all()
    return jsonify([s.to_dict() for s in shares])

@tags_shares_bp.route('/api/share/<int:share_id>', methods=['PUT'])
@login_required
def update_share(share_id):
    share = Share.query.filter_by(id=share_id, user_id=current_user.id).first_or_404()
    data = request.json or {}
    if 'share_summary' in data: share.share_summary = data['share_summary']
    if 'share_notes' in data: share.share_notes = data['share_notes']
    db.session.commit()
    return jsonify({'success': True, 'share': share.to_dict()})

@tags_shares_bp.route('/api/share/<int:share_id>', methods=['DELETE'])
@login_required
def delete_share(share_id):
    share = Share.query.filter_by(id=share_id, user_id=current_user.id).first_or_404()
    db.session.delete(share); db.session.commit()
    return jsonify({'success': True})


# --- User Preferences ---
@tags_shares_bp.route('/api/user/preferences', methods=['POST'])
@login_required
def save_user_preferences():
    data = request.json or {}
    if 'language' in data: current_user.ui_language = data['language']
    db.session.commit()
    return jsonify({'success': True, 'message': 'Preferences saved successfully', 'ui_language': current_user.ui_language})


# --- System Info ---
@tags_shares_bp.route('/api/system/info', methods=['GET'])
def get_system_info():
    try:
        from flask import current_app as app
        from src.app import get_version  # temporary until factory introduced
        version = get_version()
        from src.app import TEXT_MODEL_BASE_URL, TEXT_MODEL_NAME, USE_ASR_ENDPOINT, ASR_BASE_URL, transcription_base_url
        return jsonify({
            'version': version,
            'llm_endpoint': TEXT_MODEL_BASE_URL,
            'llm_model': TEXT_MODEL_NAME,
            'whisper_endpoint': os.environ.get('TRANSCRIPTION_BASE_URL', 'https://api.openai.com/v1'),
            'asr_enabled': USE_ASR_ENDPOINT,
            'asr_endpoint': ASR_BASE_URL if USE_ASR_ENDPOINT else None
        })
    except Exception as e:
        return jsonify({'error': 'Unable to retrieve system information'}), 500


# --- Tag Routes ---
@tags_shares_bp.route('/api/tags', methods=['GET'])
@login_required
def get_tags():
    tags = Tag.query.filter_by(user_id=current_user.id).order_by(Tag.name).all()
    return jsonify([t.to_dict() for t in tags])

@tags_shares_bp.route('/api/tags', methods=['POST'])
@login_required
def create_tag():
    data = request.get_json() or {}
    if not data.get('name'): return jsonify({'error': 'Tag name is required'}), 400
    existing = Tag.query.filter_by(name=data['name'], user_id=current_user.id).first()
    if existing: return jsonify({'error': 'Tag with this name already exists'}), 400
    tag = Tag(name=data['name'], user_id=current_user.id, color=data.get('color', '#3B82F6'), custom_prompt=data.get('custom_prompt'), default_language=data.get('default_language'), default_min_speakers=data.get('default_min_speakers'), default_max_speakers=data.get('default_max_speakers'))
    db.session.add(tag); db.session.commit(); return jsonify(tag.to_dict()), 201

@tags_shares_bp.route('/api/tags/<int:tag_id>', methods=['PUT'])
@login_required
def update_tag(tag_id):
    tag = Tag.query.filter_by(id=tag_id, user_id=current_user.id).first_or_404()
    data = request.get_json() or {}
    if 'name' in data:
        conflict = Tag.query.filter_by(name=data['name'], user_id=current_user.id).filter(Tag.id != tag_id).first()
        if conflict: return jsonify({'error': 'Another tag with this name already exists'}), 400
        tag.name = data['name']
    if 'color' in data: tag.color = data['color']
    if 'custom_prompt' in data: tag.custom_prompt = data['custom_prompt']
    if 'default_language' in data: tag.default_language = data['default_language']
    if 'default_min_speakers' in data: tag.default_min_speakers = data['default_min_speakers']
    if 'default_max_speakers' in data: tag.default_max_speakers = data['default_max_speakers']
    tag.updated_at = datetime.utcnow(); db.session.commit(); return jsonify(tag.to_dict())

@tags_shares_bp.route('/api/tags/<int:tag_id>', methods=['DELETE'])
@login_required
def delete_tag(tag_id):
    tag = Tag.query.filter_by(id=tag_id, user_id=current_user.id).first_or_404(); db.session.delete(tag); db.session.commit(); return jsonify({'success': True})

@tags_shares_bp.route('/api/recordings/<int:recording_id>/tags', methods=['POST'])
@login_required
def add_tag_to_recording(recording_id):
    recording = Recording.query.filter_by(id=recording_id, user_id=current_user.id).first_or_404()
    data = request.get_json() or {}; tag_id = data.get('tag_id')
    if not tag_id: return jsonify({'error': 'tag_id is required'}), 400
    tag = Tag.query.filter_by(id=tag_id, user_id=current_user.id).first_or_404()
    existing = RecordingTag.query.filter_by(recording_id=recording_id, tag_id=tag_id).first()
    if not existing:
        max_order = db.session.query(db.func.max(RecordingTag.order)).filter_by(recording_id=recording_id).scalar() or 0
        assoc = RecordingTag(recording_id=recording_id, tag_id=tag_id, order=max_order + 1, added_at=datetime.utcnow())
        db.session.add(assoc); db.session.commit()
    return jsonify({'success': True, 'tags': [t.to_dict() for t in recording.tags]})

@tags_shares_bp.route('/api/recordings/<int:recording_id>/tags/<int:tag_id>', methods=['DELETE'])
@login_required
def remove_tag_from_recording(recording_id, tag_id):
    recording = Recording.query.filter_by(id=recording_id, user_id=current_user.id).first_or_404(); tag = Tag.query.filter_by(id=tag_id, user_id=current_user.id).first_or_404(); assoc = RecordingTag.query.filter_by(recording_id=recording_id, tag_id=tag_id).first();
    if assoc: db.session.delete(assoc); db.session.commit()
    return jsonify({'success': True, 'tags': [t.to_dict() for t in recording.tags]})


# --- Transcript Templates ---
@tags_shares_bp.route('/api/transcript-templates', methods=['GET'])
@login_required
def get_transcript_templates():
    templates = TranscriptTemplate.query.filter_by(user_id=current_user.id).order_by(TranscriptTemplate.is_default.desc(), TranscriptTemplate.name.asc()).all()
    return jsonify([t.to_dict() for t in templates])

@tags_shares_bp.route('/api/transcript-templates', methods=['POST'])
@login_required
def create_transcript_template():
    data = request.get_json() or {}
    name = data.get('name'); template_content = data.get('template')
    if not name or not template_content: return jsonify({'error': 'Name and template are required'}), 400
    is_default = data.get('is_default', False)
    if is_default:
        TranscriptTemplate.query.filter_by(user_id=current_user.id, is_default=True).update({'is_default': False})
    tmpl = TranscriptTemplate(user_id=current_user.id, name=name, template=template_content, description=data.get('description'), is_default=is_default, created_at=datetime.utcnow(), updated_at=datetime.utcnow())
    db.session.add(tmpl); db.session.commit(); return jsonify(tmpl.to_dict()), 201

@tags_shares_bp.route('/api/transcript-templates/<int:template_id>', methods=['PUT'])
@login_required
def update_transcript_template(template_id):
    tmpl = TranscriptTemplate.query.filter_by(id=template_id, user_id=current_user.id).first_or_404(); data = request.get_json() or {}
    if 'name' in data: tmpl.name = data['name']
    if 'template' in data: tmpl.template = data['template']
    if 'description' in data: tmpl.description = data['description']
    if 'is_default' in data and data['is_default']:
        TranscriptTemplate.query.filter_by(user_id=current_user.id, is_default=True).update({'is_default': False})
        tmpl.is_default = True
    tmpl.updated_at = datetime.utcnow(); db.session.commit(); return jsonify(tmpl.to_dict())

@tags_shares_bp.route('/api/transcript-templates/<int:template_id>', methods=['DELETE'])
@login_required
def delete_transcript_template(template_id):
    tmpl = TranscriptTemplate.query.filter_by(id=template_id, user_id=current_user.id).first_or_404(); db.session.delete(tmpl); db.session.commit(); return jsonify({'success': True})

@tags_shares_bp.route('/api/transcript-templates/create-defaults', methods=['POST'])
@login_required
def create_default_templates():
    created = 0
    existing_default = TranscriptTemplate.query.filter_by(user_id=current_user.id, is_default=True).first()
    if not existing_default:
        default = TranscriptTemplate(user_id=current_user.id, name='Default', template='[{{speaker}}]: {{text}}', description='Basic transcript format', is_default=True, created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        db.session.add(default); created += 1
    db.session.commit(); return jsonify({'success': True, 'created': created})


# --- Transcript Templates Guide ---
@tags_shares_bp.route('/docs/transcript-templates-guide')
def transcript_templates_guide():
    import markdown
    docs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'docs', 'transcript-templates-guide.md')
    docs_path = os.path.abspath(docs_path)
    if not os.path.exists(docs_path): return 'Documentation not found', 404
    with open(docs_path, 'r', encoding='utf-8') as f: content = f.read()
    html = markdown.markdown(content, extensions=['extra','tables','fenced_code'])
    return html
