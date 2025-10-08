import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import create_app
from src.extensions import db
from src.models import User

app = create_app()

with app.app_context():
    users = User.query.all()
    print(f'Found {len(users)} users')
    for u in users:
        print(f'  - {u.username} ({u.email}) - Admin: {u.is_admin}')
