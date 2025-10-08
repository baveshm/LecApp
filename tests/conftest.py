import os
import tempfile
import threading
import pytest

# Ensure test-friendly env BEFORE importing app
os.environ.setdefault('ENABLE_INQUIRE_MODE', 'false')  # disable heavy features by default
os.environ.setdefault('USE_ASR_ENDPOINT', 'false')
os.environ.setdefault('TRANSCRIPTION_BASE_URL', 'http://localhost/dummy')
os.environ.setdefault('TRANSCRIPTION_API_KEY', 'dummy-key')

from src.app import app  # noqa: E402
from src.extensions import db, bcrypt  # noqa: E402
from src.models import User, Recording, Tag, Event, Share, Attachment, Speaker  # noqa: E402


@pytest.fixture(scope='session')
def test_upload_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture(scope='session', autouse=True)
def configure_app(test_upload_dir):
    # Adjust runtime config for tests
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['UPLOAD_FOLDER'] = test_upload_dir
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(test_upload_dir, 'test.db')
    # Recreate database
    with app.app_context():
        db.drop_all()
        db.create_all()
    return app


@pytest.fixture(scope='function')
def session():
    """Provide a clean database session per test."""
    with app.app_context():
        # Truncate tables (simple approach for small schema)
        for tbl in reversed(db.metadata.sorted_tables):
            db.session.execute(tbl.delete())
        db.session.commit()
        yield db.session
        db.session.rollback()


@pytest.fixture
def client(configure_app, session):
    return app.test_client()


@pytest.fixture
def user(session):
    # Create a user with a properly hashed password compatible with app login
    hashed = bcrypt.generate_password_hash('TestPass123!').decode('utf-8')
    u = User(username='tester', email='tester@example.com', password=hashed)
    session.add(u)
    session.commit()
    return u


@pytest.fixture
def auth_client(client, user):
    # Perform login
    client.post('/login', data={'email': user.email, 'password': 'TestPass123!'}, follow_redirects=True)
    return client


@pytest.fixture
def mock_transcription(monkeypatch, session):
    """Monkeypatch transcription to avoid external API and background threads.

    Strategy: patch transcribe_audio_task to immediately set the recording to COMPLETED.
    Avoid replacing threading.Thread globally (keeps rate limiter timers functional).
    """
    def fake_transcribe(app_context, recording_id, filepath, filename_for_asr, start_time, *args, **kwargs):
        with app_context:
            rec = db.session.get(Recording, recording_id)
            if rec:
                rec.status = 'COMPLETED'
                rec.transcription = '[{"speaker":"SPEAKER_00","sentence":"Hello","start_time":0,"end_time":1}]'
                db.session.commit()
    monkeypatch.setattr('src.services.transcription_service.transcribe_audio_task', fake_transcribe, raising=False)

    # Patch only the Thread class referenced in blueprints.recordings (not global threading / Timer)
    import threading as _threading
    class ImmediateThread(_threading.Thread):
        def start(self):  # run synchronously
            self.run()
    monkeypatch.setattr('src.blueprints.recordings.threading.Thread', ImmediateThread, raising=False)

    # Prevent network calls by patching OpenAI client's transcription create function
    def fake_create(*args, **kwargs):
        class R:
            text = 'Speaker 0: Hello.'
        return R()
    monkeypatch.setattr('openai.resources.audio.transcriptions.Transcriptions.create', fake_create, raising=False)
    return True
