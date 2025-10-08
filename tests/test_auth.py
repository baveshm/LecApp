def test_register_login_logout_flow(client):
    # Register
    resp = client.post('/register', data={
        'username': 'newuser',
        'email': 'new@example.com',
        'password': 'NewPass123!',
        'confirm_password': 'NewPass123!'
    }, follow_redirects=True)
    assert resp.status_code == 200
    # Login
    resp = client.post('/login', data={'email': 'new@example.com', 'password': 'NewPass123!'}, follow_redirects=True)
    assert resp.status_code == 200
    # Access account page
    account = client.get('/account', follow_redirects=True)
    assert account.status_code == 200
    # Logout
    out = client.get('/logout', follow_redirects=True)
    assert out.status_code == 200
