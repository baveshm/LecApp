from src.extensions import db
from src.models import Recording, Event
import io
from datetime import datetime


def test_events_and_ics(auth_client, mock_transcription):
    # Upload recording
    data = {'file': (io.BytesIO(b'RIFF....WAVEfmt '), 'event.wav')}
    rec_id = auth_client.post('/upload', data=data, content_type='multipart/form-data').get_json()['recording_id']
    # Insert an event directly
    from src.app import app
    with app.app_context():
        e = Event(
            recording_id=rec_id,
            title='Planning Meeting',
            start_datetime=datetime.fromisoformat('2025-10-06T10:00:00'),
            end_datetime=datetime.fromisoformat('2025-10-06T11:00:00')
        )
        db.session.add(e)
        db.session.commit()
        event_id = e.id
    # List events
    ev_list = auth_client.get(f'/api/recording/{rec_id}/events')
    assert ev_list.status_code == 200
    # Single ICS
    ics_single = auth_client.get(f'/api/event/{event_id}/ics')
    assert ics_single.status_code == 200
    assert 'BEGIN:VCALENDAR' in ics_single.get_data(as_text=True)
    # All ICS
    ics_all = auth_client.get(f'/api/recording/{rec_id}/events/ics')
    assert ics_all.status_code == 200
