import io
from src.models import Recording


def test_upload_creates_recording(auth_client, mock_transcription):
    # Create dummy audio bytes (not real audio but sufficient for path save)
    data = {
        'file': (io.BytesIO(b'RIFF....WAVEfmt '), 'sample.wav')
    }
    resp = auth_client.post('/upload', data=data, content_type='multipart/form-data')
    assert resp.status_code in (200, 202), resp.get_data(as_text=True)
    json_data = resp.get_json() or {}
    rec_id = json_data.get('recording_id') or json_data.get('id')
    assert rec_id, f"Upload response missing recording id: {json_data}"
    # Fetch status endpoint
    status_resp = auth_client.get(f'/status/{rec_id}')
    assert status_resp.status_code == 200
    status_json = status_resp.get_json()
    assert status_json['id'] == rec_id
    # Ensure transcription was mocked
    # (Recording may already be completed due to ImmediateThread patch)
    # Retrieve via internal model for certainty
    from src.extensions import db
    r = db.session.get(Recording, rec_id)
    assert r is not None
    assert r.transcription is not None


def test_toggle_flags(auth_client, mock_transcription, session):
    # Upload a recording first
    import io
    data = {'file': (io.BytesIO(b'RIFF....WAVEfmt '), 'flag.wav')}
    rec_json = auth_client.post('/upload', data=data, content_type='multipart/form-data').get_json() or {}
    rec_id = rec_json.get('recording_id') or rec_json.get('id')
    assert rec_id, f"Upload response missing recording id: {rec_json}"
    
    # Get recording from database to check initial state
    from src.extensions import db
    rec = db.session.get(Recording, rec_id)
    assert rec is not None
    
    # Check initial state
    assert rec.is_inbox is True  # Default should be True
    assert rec.is_highlighted is False  # Default should be False
    
    # Toggle inbox
    inbox_resp = auth_client.post(f'/recording/{rec_id}/toggle_inbox')
    assert inbox_resp.status_code == 200
    
    # Verify inbox state changed in database
    session.refresh(rec)
    assert rec.is_inbox is False
    
    # Toggle inbox back
    inbox_resp = auth_client.post(f'/recording/{rec_id}/toggle_inbox')
    assert inbox_resp.status_code == 200
    
    # Verify inbox state changed back
    session.refresh(rec)
    assert rec.is_inbox is True
    
    # Toggle highlight
    hl_resp = auth_client.post(f'/recording/{rec_id}/toggle_highlight')
    assert hl_resp.status_code == 200
    
    # Verify highlight state changed in database
    session.refresh(rec)
    assert rec.is_highlighted is True
    
    # Toggle highlight back
    hl_resp = auth_client.post(f'/recording/{rec_id}/toggle_highlight')
    assert hl_resp.status_code == 200
    
    # Verify highlight state changed back
    session.refresh(rec)
    assert rec.is_highlighted is False
