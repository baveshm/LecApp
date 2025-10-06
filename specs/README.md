# Specs: Attach Supporting Documents — Activity Log

This document captures the UI/UX design, code changes, MCP snapshot anchors (UIDs), runtime actions, and pending work for the new “Attach documents” feature. It is intended to survive fresh sessions and serve as the single source of truth for the work completed so far.

Index
- Context and goals
- Frontend changes (template)
- JavaScript wiring (state + handlers)
- MCP snapshots and stable UIDs
- Server lifecycle actions
- Design decisions and rationale
- Pending backend work
- Next steps and acceptance criteria
- Reproduction checklist
- Changelog (chronological)

## Context and goals
- Do not break existing frontend flows.
- Design frontend elements first; backend wiring later.
- Provide a compact “Attach” control beside Stop Recording and also after recording (before upload).
- No new heavy UI for attachments; only a concise text summary under “Estimated size” while recording.
- Support multi-select for common document/image types.
- Capture MCP snapshots for every state and use UIDs for precise placement instructions.

## Frontend changes (template)
File: [templates.index_html()](templates/index.html:710)

Key insertions and locations:
- Recording timer section (estimated size)
  - “Estimated size” line anchor: [templates.index_html()](templates/index.html:679)
  - Files summary text appears below this line when attachments exist:
    - [templates.index_html()](templates/index.html:691)
    - Renders as: “Files attached: file1.txt; file2.md; …” driven by documentUploads
- Recording action row (during recording)
  - Stop Recording button anchor: [templates.index_html()](templates/index.html:714)
  - New Attach button (paperclip + “Attach”) immediately to the right:
    - [templates.index_html()](templates/index.html:724)
    - Hidden file input (multi-select) with accept list + @change:
      - [templates.index_html()](templates/index.html:733)
- Post-recording action area (before upload)
  - Attach button also rendered for parity:
    - [templates.index_html()](templates/index.html:769)

Behavior captured in template:
- Button text: “Attach” with paperclip icon to match visual language.
- During recording and after recording prior to upload.
- Single, minimal “Files attached:” text summary shown under “Estimated size” only (no new lists/panels).
- File input accepts: .pdf .doc .docx .txt .md .rtf .ppt .pptx .xls .xlsx .csv .json .png .jpg .jpeg .heic .webp (multi-select).

## JavaScript wiring (state + handlers)
File: [static.js.app_js()](static/js/app.js:95)

Added state:
- documentUploads (ref) to reflect selected supporting documents:
  - [static.js.app_js()](static/js/app.js:95)

Added handler:
- handleDocumentSelect (frontend-only; reads FileList, updates documentUploads):
  - [static.js.app_js()](static/js/app.js:2031)
  - Scheme per item: { clientId, file, status: 'selected' | 'uploading' | 'completed' | 'failed', progress }

Exposed to template:
- documentUploads and handleDocumentSelect are returned from setup(), enabling:
  - Template v-if binding for “Files attached…” text under estimated size.
  - File input @change hook to populate selections.

Notes:
- No network upload performed yet (per “frontend first” scope).
- Placement and semantics are stable and decoupled from backend.

## MCP snapshots and stable UIDs
Recording (live) anchors:
- Stop Recording: ref=e163 (source: [templates.index_html()](templates/index.html:714))
- Attach: ref=e165 (source: [templates.index_html()](templates/index.html:724))
- Estimated size: ref=e176 (source: [templates.index_html()](templates/index.html:679))

Post-recording (before upload) anchors:
- Attach: ref=e182 (source: [templates.index_html()](templates/index.html:769))
- Tags section header: ref=e187 (context: [templates.index_html()](templates/index.html:762))
- Upload Recording & Notes: ref=e193 (context: [templates.index_html()](templates/index.html:901))
- Discard: ref=e196 (context: [templates.index_html()](templates/index.html:905))

Usage
- Use these refs to specify exact positioning/behavior adjustments in future instructions.
- We can re-snapshot any step to refresh refs if DOM changes meaningfully.

## Server lifecycle actions
Background start (nohup + disown) with logs; created data dirs:
- Command executed (virtualenv assumed already created):
  - mkdir -p ./tmp uploads instance
  - source .venv/bin/activate
  - export PYTHONPATH="$(pwd)"
  - export FLASK_APP=src.app
  - nohup flask run --host 0.0.0.0 --port 8899 > ./tmp/flask-YYYYMMDD-HHMMSS.log 2>&1 & disown
- Health checks:
  - Opened http://localhost:8899 and verified UI + console logs.
  - Observed typical dev warnings (Tailwind CDN prod note, Vue dev build).

## Design decisions and rationale
- Placement: Attach control sits in the primary action row next to Stop to keep proximity with the recording action; mirrored after recording to avoid mode switch surprises.
- Minimal surface: Only a compact button and a one-line summary under “Estimated size” (you requested no new attachment panels).
- Visibility: During recording and post-recording to cover both live capture and last-minute additions before upload.
- Non-invasive styling: Reuses existing palette classes to avoid CSS churn.
- i18n placement: Labels currently literal; can be localized once copy is finalized (we already use t(...) elsewhere).

## Pending backend work (not implemented yet)
- Decide endpoint(s) for document ingestion and recording association:
  - Option A: POST /recording/:id/documents (associate to in-progress or staged recording)
  - Option B: POST /documents with recording_id form field
- Storage & limits:
  - Enforce per-file size and total size limits aligned with existing upload constraints.
- Background processing:
  - Decide whether to reuse existing uploadQueue UI or keep the minimal text-only approach for supporting docs.
- Metadata:
  - Persist filenames and surface minimal metadata in detail view if needed (out of scope for current UI per spec).

## Next steps and acceptance criteria
- Frontend UX adjustments (if requested via MCP):
  - Any spacing tweaks relative to refs e163/e165/e176/e182.
- Backend integration:
  - Endpoint implemented; handler returns per-file status.
  - Client shows progress (optional) or simply transitions to “Done” states in a compact way.
- Acceptance:
  - Attach appears in both states as designed.
  - “Files attached:” summary displays selected filenames during recording.
  - No regressions in existing recording/upload workflows.

## Reproduction checklist
- Start server (background): see “Server lifecycle actions”.
- Navigate to http://localhost:8899
- Click “Microphone” to begin recording.
- Verify:
  - Stop Recording (ref=e163) and Attach (ref=e165) visible in one row.
  - Estimated size (ref=e176) present.
  - Choose files via Attach; observe “Files attached:” summary under estimated size.
- Stop Recording to enter post-recording view.
- Verify Attach (ref=e182), Tags (ref=e187), Upload (ref=e193), Discard (ref=e196).

## Changelog
- 2025-10-06 (local): Inserted Attach UI and files summary in [templates.index_html()](templates/index.html:710); added JS state/handler in [static.js.app_js()](static/js/app.js:95) and [static.js.app_js()](static/js/app.js:2031).
- 2025-10-06: Restarted Flask via nohup+disown; verified logs under ./tmp/.
- 2025-10-06: Captured MCP snapshots for recording and post-recording; documented UIDs.
- 2025-10-06: Confirmed spec alignment with requested placement, style, visibility, and minimal display.

## Attached Files Tab Feature Investigation (2025-10-06)

### Context
User requested to add a third tab called "Attached Files" in the Summary and Notes section of the recording detail view. The tab should display attached documents/files, provide preview capabilities, and include options to delete and unlink files.

### Investigation Results

#### Frontend Status: ALREADY IMPLEMENTED ✅
The "Attached Files" tab UI is fully implemented in the frontend:

**Tab Button Location**: [templates/index.html](templates/index.html:1560-1569)
- Tab navigation includes "Attached Files" button with paperclip icon
- Tab switching via `selectedTab = 'attached_files'`
- Positioned between "Notes" and "Events" tabs

**Tab Content Section**: [templates/index.html](templates/index.html:1704-1788)
Complete implementation includes:
- Header with refresh button
- Loading state indicator
- Empty state message
- File list with cards showing:
  - File icon (dynamic based on file type)
  - Original filename (truncated with tooltip)
  - File size and upload date
  - File type badge
  - Action buttons: Preview, Download, Delete

**JavaScript Integration**: [static/js/app.js](static/js/app.js)
Frontend code attempts to call:
- `loadAttachedFiles()` - fetches files for current recording
- `previewAttachedFile(file)` - shows preview or downloads
- `downloadAttachedFile(file)` - downloads specific file
- `deleteAttachedFile(file)` - deletes file from recording
- Auto-loads files when recording is selected via watcher

#### Backend Status: NOT IMPLEMENTED ❌
The backend infrastructure is completely missing, causing 404 errors:

**Missing Database Model:**
- No `Attachment` or `RecordingFile` model exists
- Need foreign key relationship to Recording model
- Should track: recording_id, user_id, file_path, original_filename, file_size, file_type, uploaded_at

**Missing API Endpoints:**
1. `GET /api/recordings/<int:recording_id>/files`
   - List all files attached to a recording
   - Returns JSON array of file metadata

2. `POST /api/recordings/<int:recording_id>/files`
   - Upload/attach a file to a recording
   - Handle file storage and database entry creation

3. `GET /api/recordings/<int:recording_id>/files/<int:file_id>/download`
   - Serve file for download
   - Include proper Content-Disposition headers

4. `DELETE /api/recordings/<int:recording_id>/files/<int:file_id>`
   - Delete file from storage and database
   - Verify user permissions

**Console Errors Observed (via MCP):**
```
Failed to load resource: the server responded with a status of 404 (NOT FOUND)
Error loading attached files: Error: Failed to load attached files
```

### MCP Browser Snapshots (Desktop View)

**Recording Detail View - Tab Navigation** (ref=e180):
- Summary tab button: ref=e181
- Notes tab button: ref=e182
- **Attached Files tab button would be ref=e183** (not captured in current snapshot as it's already in HTML but needs backend)
- Events tab button (conditional): ref=e184

**Current Visible Elements:**
- Recording title heading: ref=e118
- Action buttons row: ref=e119-e135
- Participant info: ref=e139
- Meeting date: ref=e142
- Transcription section: ref=e153-e173
- Audio player: ref=e178
- Tab navigation container: ref=e179

### Related Existing Features

**Supporting Documents During Recording** (from specs):
- Attach button during recording: ref=e165 ([templates/index.html](templates/index.html:724))
- Attach button after recording: ref=e182 ([templates/index.html](templates/index.html:769))
- documentUploads state: [static/js/app.js](static/js/app.js:95)
- handleDocumentSelect handler: [static/js/app.js](static/js/app.js:2031)

**Note:** The recording-time attachment feature is also frontend-only currently. Both features need the same backend infrastructure.

### Architecture Design Decisions

**Database Schema (Proposed):**
```python
class Attachment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    recording_id = db.Column(db.Integer, db.ForeignKey('recording.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    original_filename = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    file_type = db.Column(db.String(50), nullable=True)  # pdf, docx, png, etc.
    mime_type = db.Column(db.String(100), nullable=True)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    recording = db.relationship('Recording', backref=db.backref('attachments', lazy=True, cascade='all, delete-orphan'))
    user = db.relationship('User', backref=db.backref('attachments', lazy=True, cascade='all, delete-orphan'))
```

**Storage Strategy:**
- Store attachments in same `UPLOAD_FOLDER` with subdirectory structure
- Pattern: `{UPLOAD_FOLDER}/attachments/{recording_id}/{filename}`
- Reuse existing file size validation from SystemSetting (max_file_size_mb)

**Security Considerations:**
- Verify user owns recording before allowing file operations
- Use `secure_filename()` for all uploaded files
- Validate file extensions against allowed types
- Apply same permission checks as recording access

### Implementation Checklist

**Phase 1: Database Model** ✅ (Completed 2025-10-06)
- [x] Create Attachment model in src/app.py
- [x] Add migration logic for new table
- [x] Add to_dict() method for JSON serialization
- [x] Test database creation

**Implementation Details:**
- Model location: [`src/app.py`](src/app.py:1527-1552)
- Migration script: [`scripts/migrate_attachments.py`](scripts/migrate_attachments.py)
- Database table: `attachment` with indexes on `recording_id` and `user_id`
- Fields: id, recording_id, user_id, file_path, original_filename, file_size, file_type, mime_type, uploaded_at
- Relationships: Foreign keys to Recording and User with cascade delete
- JSON serialization: `to_dict()` method returns all fields in API-friendly format

**Phase 2: API Endpoints** (Not started)
- [ ] GET /api/recordings/<id>/files - list files
- [ ] POST /api/recordings/<id>/files - upload file
- [ ] GET /api/recordings/<id>/files/<file_id>/download - serve file
- [ ] DELETE /api/recordings/<id>/files/<file_id> - delete file
- [ ] Add proper error handling and logging

**Phase 3: Integration** (Not started)
- [ ] Update upload flow to handle documentUploads array
- [ ] Store files during recording upload
- [ ] Handle file cleanup on recording deletion
- [ ] Update Recording.to_dict() to include attachment count

**Phase 4: Testing** (Not started)
- [ ] Test file upload during recording
- [ ] Test file upload after recording
- [ ] Test file download
- [ ] Test file deletion
- [ ] Test permission verification
- [ ] Verify MCP browser shows files correctly

### Next Steps

1. Create Attachment database model in [`src/app.py`](src/app.py) after existing models (around line 1500)
2. Add API endpoints for file management (after line 6500 in [`src/app.py`](src/app.py))
3. Update [`uploadRecordedAudio()`](static/js/app.js) to handle documentUploads
4. Test complete flow via MCP browser
5. Update this specs document with implementation details

### Current Blocker

Backend implementation is completely missing. Frontend is production-ready and waiting for API endpoints to be created.

### File References

**Frontend Files:**
- Tab UI: [templates/index.html](templates/index.html:1560-1788)
- Tab button: line 1560-1569
- Tab content: line 1704-1788
- JavaScript state: [static/js/app.js](static/js/app.js:95)
- File handlers: Would be in [static/js/app.js](static/js/app.js) (need to verify exact locations)

**Backend Files:**
- Models: [src/app.py](src/app.py:1214-1526) (existing models section)
- Endpoints: [src/app.py](src/app.py:6249) (upload_file exists, need attachment endpoints)
- File utilities: [src/app.py](src/app.py:2813-2868) (extract_audio_from_video for reference)

### Session Summary

**What Was Done:**
1. Read and analyzed specs/README.md to understand existing "Attach documents" feature
2. Used MCP browser to navigate to application (http://localhost:8899)
3. Logged in with credentials (dev@example.com)
4. Navigated to recording detail view
5. Captured MCP snapshot of current page state
6. Discovered "Attached Files" tab already exists in frontend
7. Identified missing backend infrastructure
8. Documented findings in this section

**Time Spent:** Initial investigation and documentation
**Status:** Ready for backend implementation phase
**Next Agent:** Should implement Attachment model and API endpoints as outlined above

End of Attached Files Tab Investigation.

---

## Phase 1 Implementation Summary (2025-10-06)

### Completed Work

**Database Model Created** ✅
- Added [`Attachment`](src/app.py:1527-1552) model class to [`src/app.py`](src/app.py)
- Fields implemented:
  - `id`: Primary key
  - `recording_id`: Foreign key to Recording (with cascade delete)
  - `user_id`: Foreign key to User (with cascade delete)
  - `file_path`: Storage path for the file
  - `original_filename`: User's original filename
  - `file_size`: File size in bytes
  - `file_type`: File extension (pdf, docx, png, etc.)
  - `mime_type`: MIME type for proper serving
  - `uploaded_at`: Timestamp of upload
- Relationships configured with cascade delete on both Recording and User
- `to_dict()` method implemented for JSON API responses

**Migration Script Created** ✅
- Created [`scripts/migrate_attachments.py`](scripts/migrate_attachments.py)
- Script creates `attachment` table with proper foreign keys
- Adds indexes on `recording_id` and `user_id` for query performance
- Includes table existence check to prevent duplicate migrations
- Successfully executed - database table created

**Database Migration Executed** ✅
- Ran migration script successfully
- Table `attachment` created in database
- Indexes created for optimal query performance
- Verified table structure matches model definition

### Next Steps

**Phase 2: File Upload Endpoint** (Ready to implement)
- Implement `POST /api/recordings/<int:recording_id>/files`
- Handle multipart file uploads
- Validate file types and sizes
- Store files in `{UPLOAD_FOLDER}/attachments/{recording_id}/` structure
- Create Attachment database records

**Phase 3: File List Endpoint** (Ready to implement)
- Implement `GET /api/recordings/<int:recording_id>/files`
- Return JSON array of attachment metadata
- Verify user permissions

**Phase 4: File Preview Endpoint** (Ready to implement)
- Implement `GET /api/recordings/<int:recording_id>/files/<int:file_id>/preview`
- Generate previews for first 2 pages/sections
- Support multiple file types (images, PDFs, documents, spreadsheets)

**Phase 5: File Delete Endpoint** (Ready to implement)
- Implement `DELETE /api/recordings/<int:recording_id>/files/<int:file_id>`
- Remove files from storage and database
- Verify user permissions

### Technical Notes

**Storage Strategy:**
- Files stored in: `{UPLOAD_FOLDER}/attachments/{recording_id}/{secure_filename}`
- Reuses existing `UPLOAD_FOLDER` configuration
- Automatic directory creation on upload

**Security:**
- User ownership verification on all operations
- `secure_filename()` applied to all uploads
- File type validation against allowed extensions
- Size limits enforced via existing SystemSetting

**Allowed File Types:**
- Documents: .pdf, .doc, .docx, .txt, .md, .rtf
- Presentations: .ppt, .pptx
- Spreadsheets: .xls, .xlsx, .csv
- Data: .json
- Images: .png, .jpg, .jpeg, .heic, .webp

---