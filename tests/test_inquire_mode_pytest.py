"""
Test suite for Inquire Mode functionality using pytest framework.
"""
import pytest
from src.app import app
from src.extensions import db
from src.models import User, Recording, TranscriptChunk, InquireSession


def test_database_models_exist(session):
    """Test that the new database models work correctly."""
    with app.app_context():
        # Test that tables exist
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        
        required_tables = ['transcript_chunk', 'inquire_session']
        for table in required_tables:
            assert table in tables, f"Table '{table}' missing"


def test_transcript_chunk_creation(session, auth_client, mock_transcription):
    """Test TranscriptChunk creation and relationships."""
    # Create a test recording first
    import io
    data = {'file': (io.BytesIO(b'RIFF....WAVEfmt '), 'chunk_test.wav')}
    rec_json = auth_client.post('/upload', data=data, content_type='multipart/form-data').get_json() or {}
    rec_id = rec_json.get('recording_id') or rec_json.get('id')
    assert rec_id, "Upload response missing recording id"
    
    # Get user and recording
    user = User.query.first()
    assert user is not None, "No user found for testing"
    recording = db.session.get(Recording, rec_id)
    assert recording is not None, "Recording not found after upload"
    
    # Test TranscriptChunk creation
    chunk = TranscriptChunk()
    chunk.recording_id = recording.id
    chunk.user_id = user.id
    chunk.chunk_index = 0
    chunk.content = "This is a test transcription chunk."
    chunk.start_time = 0.0
    chunk.end_time = 5.0
    chunk.speaker_name = "Test Speaker"
    
    db.session.add(chunk)
    db.session.commit()
    
    # Verify chunk was created
    assert chunk.id is not None
    assert chunk.recording_id == recording.id
    assert chunk.content == "This is a test transcription chunk."
    
    # Clean up
    db.session.delete(chunk)
    db.session.commit()


def test_inquire_session_creation(session):
    """Test InquireSession creation and relationships."""
    # Create a test user if none exist
    user = User.query.first()
    if not user:
        pytest.skip("No users found for testing")
    
    # Test InquireSession creation
    session_obj = InquireSession()
    session_obj.user_id = user.id
    session_obj.session_name = "Test Session"
    session_obj.filter_tags = '[]'
    session_obj.filter_speakers = '["Test Speaker"]'
    
    db.session.add(session_obj)
    db.session.commit()
    
    # Verify session was created
    assert session_obj.id is not None
    assert session_obj.user_id == user.id
    assert session_obj.session_name == "Test Session"
    assert session_obj.filter_tags == '[]'
    assert session_obj.filter_speakers == '["Test Speaker"]'
    
    # Clean up
    db.session.delete(session_obj)
    db.session.commit()


def test_chunking_functions():
    """Test the chunking and embedding functions."""
    with app.app_context():
        from src.services.llm_service import chunk_transcription
        
        # Test chunking
        test_text = "This is a test sentence. This is another sentence for testing. And here's a third sentence to make sure chunking works properly with longer text that should be split into multiple chunks."
        chunks = chunk_transcription(test_text, max_chunk_length=100, overlap=20)
        
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        
        # Test embedding functions (will only work if sentence-transformers is installed)
        try:
            from src.services.llm_service import generate_embeddings, serialize_embedding, deserialize_embedding
            
            embeddings = generate_embeddings(["test sentence", "another test"])
            assert len(embeddings) == 2
            
            # Test serialization
            if embeddings[0] is not None:
                serialized = serialize_embedding(embeddings[0])
                deserialized = deserialize_embedding(serialized)
                assert deserialized is not None
                assert len(deserialized) > 0
                
        except ImportError:
            pytest.skip("sentence-transformers not installed")


def test_api_imports():
    """Test that all API endpoints can be imported."""
    from src.blueprints.inquire import (
        get_inquire_sessions, create_inquire_session, inquire_search, 
        inquire_chat
    )
    
    # Just verify imports work - functions should be callable
    assert callable(get_inquire_sessions)
    assert callable(create_inquire_session)
    assert callable(inquire_search)
    assert callable(inquire_chat)