"""
Security and error path tests for the application.
Tests authentication, authorization, and error handling.
"""
import pytest
from src.models import User, Recording, Tag
from src.extensions import db


class TestAuthenticationSecurity:
    """Test authentication and authorization security."""
    
    def test_unauthorized_access_to_protected_routes(self, client):
        """Test that unauthenticated users cannot access protected routes."""
        protected_routes = [
            '/',
            '/recordings',
            '/upload',
            '/admin',
            '/inquire'
        ]
        
        for route in protected_routes:
            resp = client.get(route)
            # Should redirect to login or return 401/403
            assert resp.status_code in (302, 401, 403), f"Route {route} should be protected"
    
    def test_non_admin_access_to_admin_routes(self, auth_client):
        """Test that non-admin users cannot access admin routes."""
        admin_routes = [
            '/admin',
            '/admin/users',
            '/admin/settings',
            '/admin/statistics'
        ]
        
        for route in admin_routes:
            resp = auth_client.get(route)
            # Should return 403 Forbidden
            assert resp.status_code == 403, f"Route {route} should be admin-only"
    
    def test_user_cannot_access_other_users_recordings(self, client, session):
        """Test that users cannot access recordings belonging to other users."""
        # Create two users
        user1 = User()
        user1.username = "user1"
        user1.email = "user1@test.com"
        user1.password = "password123"
        user1.is_admin = False
        
        user2 = User()
        user2.username = "user2"
        user2.email = "user2@test.com"
        user2.password = "password123"
        user2.is_admin = False
        
        db.session.add(user1)
        db.session.add(user2)
        db.session.commit()
        
        # Create a recording for user1
        recording = Recording()
        recording.user_id = user1.id
        recording.title = "User1's Recording"
        recording.transcription = "Test transcription"
        recording.status = "COMPLETED"
        db.session.add(recording)
        db.session.commit()
        
        # Login as user2 and try to access user1's recording
        client.post('/login', data={
            'email': 'user2@test.com',
            'password': 'password123'
        })
        
        # Try to access user1's recording
        resp = client.get(f'/recording/{recording.id}')
        # Should return 404 or 403
        assert resp.status_code in (404, 403)
        
        # Try to modify user1's recording
        resp = client.post(f'/recording/{recording.id}/toggle_highlight')
        assert resp.status_code in (404, 403)
        
        # Clean up
        db.session.delete(recording)
        db.session.delete(user1)
        db.session.delete(user2)
        db.session.commit()
    
    def test_registration_validation_errors(self, client):
        """Test registration form validation."""
        # Test weak password
        resp = client.post('/register', data={
            'username': 'testuser',
            'email': 'test@test.com',
            'password': 'weak',
            'confirm_password': 'weak'
        })
        assert resp.status_code == 200  # Should stay on registration page
        assert b'Password must be at least 8 characters long' in resp.data
        
        # Test mismatched passwords
        resp = client.post('/register', data={
            'username': 'testuser',
            'email': 'test@test.com',
            'password': 'StrongPassword123!',
            'confirm_password': 'DifferentPassword123!'
        })
        assert resp.status_code == 200
        assert b'Field must be equal to password' in resp.data
        
        # Test invalid email
        resp = client.post('/register', data={
            'username': 'testuser',
            'email': 'invalid-email',
            'password': 'StrongPassword123!',
            'confirm_password': 'StrongPassword123!'
        })
        assert resp.status_code == 200
        assert b'Invalid email address' in resp.data
        
        # Test duplicate username
        # First, create a user
        user = User()
        user.username = "existinguser"
        user.email = "existing@test.com"
        user.password = "StrongPassword123!"
        db.session.add(user)
        db.session.commit()
        
        # Try to register with same username
        resp = client.post('/register', data={
            'username': 'existinguser',
            'email': 'new@test.com',
            'password': 'StrongPassword123!',
            'confirm_password': 'StrongPassword123!'
        })
        assert resp.status_code == 200
        assert b'That username is already taken' in resp.data
        
        # Clean up
        db.session.delete(user)
        db.session.commit()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_404_handling(self, client):
        """Test 404 error handling."""
        resp = client.get('/nonexistent-route')
        assert resp.status_code == 404
    
    def test_invalid_recording_id(self, auth_client):
        """Test handling of invalid recording IDs."""
        # Test with non-existent recording ID
        resp = auth_client.get('/recording/99999')
        assert resp.status_code == 404
        
        # Test with invalid recording ID format
        resp = auth_client.get('/recording/invalid')
        assert resp.status_code == 404
        
        # Test toggle operations with invalid IDs
        resp = auth_client.post('/recording/99999/toggle_highlight')
        assert resp.status_code == 404
    
    def test_invalid_file_upload(self, auth_client):
        """Test handling of invalid file uploads."""
        # Test with no file
        resp = auth_client.post('/upload', data={}, content_type='multipart/form-data')
        assert resp.status_code == 400
        
        # Test with empty file
        import io
        data = {'file': (io.BytesIO(b''), 'empty.wav')}
        resp = auth_client.post('/upload', data=data, content_type='multipart/form-data')
        assert resp.status_code == 400
        
        # Test with unsupported file type
        data = {'file': (io.BytesIO(b'fake content'), 'file.txt')}
        resp = auth_client.post('/upload', data=data, content_type='multipart/form-data')
        assert resp.status_code == 400
    
    def test_tag_operations_errors(self, auth_client, session):
        """Test error handling in tag operations."""
        # Test creating tag with invalid data
        resp = auth_client.post('/tags', json={
            'name': '',  # Empty name
            'color': 'invalid-color'
        })
        assert resp.status_code == 400
        
        # Test adding non-existent tag to recording
        import io
        data = {'file': (io.BytesIO(b'RIFF....WAVEfmt '), 'test.wav')}
        rec_json = auth_client.post('/upload', data=data, content_type='multipart/form-data').get_json() or {}
        rec_id = rec_json.get('recording_id') or rec_json.get('id')
        
        resp = auth_client.post(f'/recordings/{rec_id}/tags', json={
            'tag_ids': [99999]  # Non-existent tag ID
        })
        assert resp.status_code == 404
        
        # Clean up
        recording = db.session.get(Recording, rec_id)
        if recording:
            db.session.delete(recording)
            db.session.commit()
    
    def test_share_operations_errors(self, auth_client):
        """Test error handling in share operations."""
        # Test accessing non-existent share
        resp = auth_client.get('/share/nonexistentshareid')
        assert resp.status_code == 404
        
        # Test creating share for non-existent recording
        resp = auth_client.post('/recordings/99999/share', json={
            'share_summary': True,
            'share_notes': True
        })
        assert resp.status_code == 404


class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_sql_injection_prevention(self, auth_client):
        """Test that SQL injection attempts are handled safely."""
        # This is a basic test - in practice, you'd want more comprehensive testing
        malicious_input = "'; DROP TABLE recordings; --"
        
        # Test in search (if search endpoint exists)
        resp = auth_client.get(f'/recordings?search={malicious_input}')
        # Should not crash the server
        assert resp.status_code in (200, 400)
        
        # Test in tag creation
        resp = auth_client.post('/tags', json={
            'name': malicious_input,
            'color': '#FF0000'
        })
        # Should either succeed (if properly sanitized) or fail validation
        assert resp.status_code in (200, 201, 400)
    
    def test_xss_prevention(self, auth_client):
        """Test XSS prevention in user inputs."""
        xss_payload = "<script>alert('xss')</script>"
        
        # Test in recording title/note fields
        import io
        data = {
            'file': (io.BytesIO(b'RIFF....WAVEfmt '), 'test.wav'),
            'title': xss_payload,
            'notes': xss_payload
        }
        resp = auth_client.post('/upload', data=data, content_type='multipart/form-data')
        
        if resp.status_code in (200, 202):
            # If upload succeeded, check that script tags are escaped
            rec_json = resp.get_json() or {}
            rec_id = rec_json.get('recording_id') or rec_json.get('id')
            
            resp = auth_client.get(f'/recording/{rec_id}')
            assert resp.status_code == 200
            # Should not contain the actual script tag
            assert b'<script>' not in resp.data
    
    def test_large_input_handling(self, auth_client):
        """Test handling of excessively large inputs."""
        # Test very large title
        large_title = 'A' * 10000  # Much larger than the 200 char limit
        
        import io
        data = {
            'file': (io.BytesIO(b'RIFF....WAVEfmt '), 'test.wav'),
            'title': large_title
        }
        resp = auth_client.post('/upload', data=data, content_type='multipart/form-data')
        # Should either truncate or reject
        assert resp.status_code in (200, 202, 400)