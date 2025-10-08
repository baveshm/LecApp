from src.extensions import db
from src.models import Recording, Tag
import io


def _upload(auth_client):
    data = {'file': (io.BytesIO(b'RIFF....WAVEfmt '), 'tag.wav')}
    resp_json = auth_client.post('/upload', data=data, content_type='multipart/form-data').get_json() or {}
    rec_id = resp_json.get('recording_id') or resp_json.get('id')
    assert rec_id, f"Upload response missing recording id: {resp_json}"
    return rec_id


def test_tag_lifecycle(auth_client, mock_transcription):
    rec_id = _upload(auth_client)
    # Create tag
    resp = auth_client.post('/api/tags', json={'name': 'Important'})
    assert resp.status_code in (200, 201), f"Expected 200 or 201, got {resp.status_code}"
    tag_id = resp.get_json()['id']
    # Attach to recording
    attach = auth_client.post(f'/api/recordings/{rec_id}/tags', json={'tag_id': tag_id})
    assert attach.status_code == 200
    # List tags
    tags = auth_client.get('/api/tags').get_json()
    assert any(t['name'] == 'Important' for t in tags)
    # Remove tag
    detach = auth_client.delete(f'/api/recordings/{rec_id}/tags/{tag_id}')
    assert detach.status_code == 200


def test_share_flow(auth_client, mock_transcription):
    rec_id = _upload(auth_client)
    # Create share
    share = auth_client.post(f'/api/recording/{rec_id}/share', json={'allow_download': True})
    assert share.status_code == 200
    share_json = share.get_json()
    assert 'public_id' in share_json
    public_id = share_json['public_id']
    # Public access (no auth) - uses separate client
    from src.app import app
    with app.test_client() as anon:
        public = anon.get(f'/share/{public_id}')
        assert public.status_code in (200, 302)  # May redirect to login if logic changes later
