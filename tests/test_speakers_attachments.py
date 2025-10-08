import io


def test_speaker_crud(auth_client):
    # Create
    create = auth_client.post('/speakers', json={'name': 'Alice'})
    assert create.status_code in (200, 201)
    # List
    lst = auth_client.get('/speakers')
    assert lst.status_code == 200
    assert any(s['name'] == 'Alice' for s in lst.get_json())


def test_attachment_flow(auth_client, mock_transcription):
    # Upload recording
    import io
    rec_resp = auth_client.post('/upload', data={'file': (io.BytesIO(b'RIFF....WAVEfmt '), 'att.wav')}, content_type='multipart/form-data')
    rec_json = rec_resp.get_json() or {}
    rec_id = rec_json.get('recording_id') or rec_json.get('id')
    assert rec_id, f"Upload response missing recording id keys: {rec_json}"
    # Upload attachment (simple text)
    attach = auth_client.post(f'/api/recordings/{rec_id}/files', data={'file': (io.BytesIO(b'hello'), 'notes.txt')}, content_type='multipart/form-data')
    assert attach.status_code in (200, 201)
    # List attachments
    files = auth_client.get(f'/api/recordings/{rec_id}/files').get_json()
    assert len(files) == 1
    file_id = files[0]['id']
    # Preview
    preview = auth_client.get(f'/api/recordings/{rec_id}/files/{file_id}/preview')
    assert preview.status_code == 200
    # Delete
    delete = auth_client.delete(f'/api/recordings/{rec_id}/files/{file_id}')
    assert delete.status_code == 200
