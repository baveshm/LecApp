from src.extensions import db, bcrypt
from src.models import User
from src.app import app

with app.app_context():
    # Check if admin user exists
    user = User.query.filter_by(email='admin@test.com').first()
    if user:
        print(f'User admin@test.com already exists')
    else:
        # Create admin user
        hashed_password = bcrypt.generate_password_hash('Admin123!').decode('utf-8')
        user = User(
            username='admin',
            email='admin@test.com',
            password=hashed_password,
            is_admin=True
        )
        db.session.add(user)
        db.session.commit()
        print(f'Created admin user: admin@test.com / Admin123!')
