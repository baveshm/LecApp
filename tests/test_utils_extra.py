from src.utils import sanitize_html, md_to_html, generate_ics_content
from datetime import datetime



def test_sanitize_html_removes_templates():
    dirty = '<script>alert(1)</script>{{danger}}<p>Safe</p>'
    cleaned = sanitize_html(dirty)
    assert '<script>' not in cleaned
    assert '{{danger}}' not in cleaned
    assert '<p>Safe</p>' in cleaned


def test_markdown_conversion():
    md = '# Title\n\n**Bold** text'
    html = md_to_html(md)
    assert '<h1>' in html and '<strong>' in html


def test_generate_ics_content_basic():
    class DummyEvent:
        def __init__(self):
            self.id = 1
            self.title = 'Sync'
            self.description = 'Weekly sync'
            self.start_datetime = datetime(2025, 10, 6, 10, 0, 0)
            self.end_datetime = datetime(2025, 10, 6, 11, 0, 0)
            self.location = 'Remote'
            self.attendees = None
            self.reminder_minutes = 30
    event = DummyEvent()
    ics = generate_ics_content(event)
    assert 'BEGIN:VCALENDAR' in ics and 'Weekly sync' in ics and 'SUMMARY:Sync' in ics
