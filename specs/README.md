Of course. Here is a condensed version of your `README.md`, stripped down to the essential facts of what has been implemented and where. This serves as a perfect, high-signal reference for an agent.

---

# Implementation Reference: Attach Supporting Documents & Files Tab

This document is a technical summary of the implemented features for attaching and managing supporting documents. It specifies the locations of UI elements, JavaScript state, database models, and API endpoints.

### 1. UI Implementation (Frontend)

#### **File Locations:**
-   **HTML Template:** `templates/index.html`
-   **JavaScript Logic:** `static/js/app.js`

#### **Feature 1: Attach Control (During & Post-Recording)**
-   **"Attach" Button (During Recording):**
    -   A button with a paperclip icon is placed to the right of the "Stop Recording" button.
    -   Location: `templates/index.html:724`
-   **"Attach" Button (Post-Recording):**
    -   The same button is rendered in the post-recording action area.
    -   Location: `templates/index.html:769`
-   **File Summary Display:**
    -   When files are selected, a text summary "Files attached: file1.txt; ..." appears.
    -   Location: Below the "Estimated size" line, implemented at `templates/index.html:691`.
-   **JavaScript State:**
    -   `documentUploads` (ref) holds the list of selected files.
    -   Location: `static/js/app.js:95`
-   **JavaScript Handler:**
    -   `handleDocumentSelect` processes the file input.
    -   Location: `static/js/app.js:2031`

#### **Feature 2: "Attached Files" Tab (Recording Detail View)**
-   **Tab Navigation Button:**
    -   An "Attached Files" tab button exists in the main tab navigation.
    -   Location: `templates/index.html:1560-1569`
-   **Tab Content Panel:**
    -   A complete UI for listing, previewing, downloading, and deleting attached files.
    -   Location: `templates/index.html:1704-1788`
-   **JavaScript Handlers:**
    -   Functions like `previewFile()`, `confirmDeleteAttachedFile()`, `refreshAttachedFiles()`, and `getFileIcon()` are implemented to power the tab's functionality.
    -   Location: `static/js/app.js:5156-5403`

### 2. UI Element Anchors (MCP UIDs)

-   **During Recording:**
    -   Stop Recording Button: `ref=e163`
    -   Attach Button: `ref=e165`
    -   Estimated Size Text: `ref=e176`
-   **Post-Recording (Before Upload):**
    -   Attach Button: `ref=e182`
-   **Recording Detail View (Tab Navigation):**
    -   Notes Tab: `ref=e182`
    -   Attached Files Tab (would be): `ref=e183`

### 3. Backend API Implementation

#### **File Location:**
-   **All Models & Endpoints:** `src/app.py`

#### **Database Model: `Attachment`**
-   A database model named `Attachment` has been created and migrated.
-   **Location:** `src/app.py:1527-1552`
-   **Key Fields:** `id`, `recording_id`, `user_id`, `file_path`, `original_filename`, `file_size`, `file_type`, `mime_type`, `uploaded_at`.
-   **Relationships:** Configured with cascade delete for `Recording` and `User`.

#### **API Endpoints**
-   **`GET /api/recordings/<id>/files`**
    -   **Function:** Lists all file metadata for a given recording.
    -   **Location:** `src/app.py:6601-6621`
-   **`POST /api/recordings/<id>/files`**
    -   **Function:** Handles multipart/form-data file uploads, validates file type and size, stores the file, and creates an `Attachment` record.
    -   **Location:** `src/app.py:6623-6710`
-   **`GET /api/recordings/<id>/files/<file_id>/preview`**
    -   **Function:** Generates and serves previews for various file types (Images, PDFs, Text, DOCX, XLSX, PPTX).
    -   **Location:** `src/app.py:6712-6889`
-   **`DELETE /api/recordings/<id>/files/<file_id>`**
    -   **Function:** Deletes the physical file from storage and removes the `Attachment` record from the database.
    -   **Location:** `src/app.py:6891-6929`

#### **Integration with Existing Features**
-   **Recording Deletion:** The `delete_recording()` endpoint has been updated to also delete all associated physical attachment files.
-   **Location:** `src/app.py:6585-6633`