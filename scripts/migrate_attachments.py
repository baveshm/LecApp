#!/usr/bin/env python3
"""
Migration script to add the Attachment table to existing databases.
This script creates the attachment table and sets up the necessary relationships.

Usage:
    python scripts/migrate_attachments.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
from datetime import datetime

# Load environment variables
load_dotenv()

def get_database_uri():
    """Get database URI from environment or use default."""
    return os.environ.get('SQLALCHEMY_DATABASE_URI', 'sqlite:///./instance/transcriptions.db')

def check_table_exists(engine, table_name):
    """Check if a table exists in the database."""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()

def migrate_attachments():
    """Create the attachment table if it doesn't exist."""
    database_uri = get_database_uri()
    print(f"Connecting to database: {database_uri}")
    
    engine = create_engine(database_uri)
    
    try:
        # Check if attachment table already exists
        if check_table_exists(engine, 'attachment'):
            print("✓ Attachment table already exists. No migration needed.")
            return True
        
        print("Creating attachment table...")
        
        with engine.connect() as conn:
            # Create attachment table
            conn.execute(text("""
                CREATE TABLE attachment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    file_path VARCHAR(500) NOT NULL,
                    original_filename VARCHAR(500) NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_type VARCHAR(50),
                    mime_type VARCHAR(100),
                    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (recording_id) REFERENCES recording(id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE
                )
            """))
            
            # Create indexes for better query performance
            conn.execute(text("""
                CREATE INDEX idx_attachment_recording_id ON attachment(recording_id)
            """))
            
            conn.execute(text("""
                CREATE INDEX idx_attachment_user_id ON attachment(user_id)
            """))
            
            conn.commit()
            
        print("✓ Attachment table created successfully")
        print("✓ Indexes created for recording_id and user_id")
        
        # Verify table was created
        if check_table_exists(engine, 'attachment'):
            print("✓ Migration completed successfully")
            
            # Show table structure
            inspector = inspect(engine)
            columns = inspector.get_columns('attachment')
            print("\nAttachment table structure:")
            for col in columns:
                print(f"  - {col['name']}: {col['type']}")
            
            return True
        else:
            print("✗ Error: Table was not created")
            return False
            
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        engine.dispose()

def main():
    """Main migration function."""
    print("=" * 60)
    print("Attachment Table Migration Script")
    print("=" * 60)
    print()
    
    success = migrate_attachments()
    
    print()
    print("=" * 60)
    if success:
        print("Migration completed successfully!")
        print()
        print("Next steps:")
        print("1. Restart your Flask application")
        print("2. The Attachment model is now available for use")
        print("3. You can now attach files to recordings")
    else:
        print("Migration failed. Please check the error messages above.")
        sys.exit(1)
    print("=" * 60)

if __name__ == '__main__':
    main()