# Agent Memory & Project Runbook (`AGENTS.md`)

**Instructions for Agent:** You have a limited context window. This document is your long-term memory and runbook. You **MUST** refer to it to understand the project structure, run the server, and make safe edits.

---

### **1. Core Objective**

This project is a Flask-based web application named "LecApp" for audio transcription and summarization. The primary goal is to refactor its monolithic structure into a modular one and upgrade its chat functionality from a basic RAG implementation to an intelligent, tool-using **ReAct agent** using LangChain and Qdrant.

---

### **2. CRITICAL - Refactored Project Structure**

The application is no longer a single `app.py` file. The logic is now separated into the following modules inside the `src/` directory:

*   **`src/app.py`**: **App Factory**. This is the main entry point. It contains the `create_app()` function which initializes the app, loads configuration, registers all blueprints, and contains the root routes (`/` and `/api/config`).

*   **`src/extensions.py`**: **Flask Extensions**. Initializes all third-party Flask extensions (`db` for SQLAlchemy, `bcrypt`, `limiter`, `csrf`, `login_manager`).

*   **`src/models.py`**: **Data Layer**. Defines all SQLAlchemy database models (like `User`, `Recording`, `Attachment`) and all WTForms classes (like `RegistrationForm`).

*   **`src/utils.py`**: **Utilities**. Contains self-contained helper functions that do not depend on the Flask app context (e.g., `safe_json_loads`, `md_to_html`, `process_markdown_to_docx`).

*   **`src/blueprints/`**: **Web Layer (Routes)**. This directory contains all the application's routes, organized by feature.
    *   `auth.py`: Handles user registration, login, logout, and the `/account` page.
    *   `recordings.py`: The largest blueprint. Handles the main recording list, file uploads, status polling, audio playback, transcription/summary/notes updates, speaker management, downloads, and the single-recording chat (`/chat`).
    *   `attachments.py`: Handles all API endpoints for uploading, listing, previewing, and deleting attached files (`/api/recordings/.../files`).
    *   `inquire.py`: Handles all "Inquire Mode" functionality, including the UI page (`/inquire`) and the multi-recording RAG chat (`/api/inquire/chat`).
    *   `admin.py`: Contains all routes for the admin dashboard (`/admin`).
    *   `tags_shares.py`: Contains all routes for managing tags and public share links.

*   **`src/services/`**: **Business Logic Layer**. This is where the core "work" of the application happens.
    *   `llm_service.py`: **Handles all LLM interactions.** This includes generating summaries, titles, identifying speakers, and **all logic for creating embeddings and performing semantic search**.
    *   `transcription_service.py`: **Handles all audio transcription.** This includes calling the ASR endpoint or Whisper API, audio/video conversion, and audio chunking.
    *   `vector_store_service.py`: **Handles all interaction with the Qdrant vector database.** This is the abstraction layer for vector search.
    *   `file_monitor.py`: Contains the logic for the background service that automatically processes files dropped into a watch directory.

---

### **3. How to Run the Web Server (Agent Workflow)**

**Prerequisites:** Python 3.11+, `ffmpeg`.

**Step 1: Activate Virtual Environment (CRITICAL)**
Before executing any Python command, you **MUST** run this:
```bash
source .venv/bin/activate
```

**Step 2: Install/Update Dependencies**
If you add a new package, pin it in `requirements.txt`, then run:
```bash
pip install -r requirements.txt
```

**Step 3: Start the Development Server**
To start the server in the background and log its output for later review, you **MUST** use the following command. This prevents the server from blocking your terminal.

```bash
nohup flask run --host 0.0.0.0 --port 8899 > /tmp/lecapp_server.log 2>&1 &
```

**Step 4: Verify Server is Running**
After starting the server, confirm it is running correctly.
- **Check Logs:** `tail -f /tmp/lecapp_server.log`
- **Health Check URL:** Access `http://localhost:8899/login`. You should see the login page.

**Step 5: Stop the Server**
When you are finished, find and stop the Flask process.
```bash
ps aux | grep "flask run" | grep -v grep | awk '{print $2}' | xargs kill
```

---

### **4. How to Test and Verify Changes**

You are not required to write new test scripts. Use your primary capabilities (e.g., `bash`, `curl`, browser interaction) to verify changes.

**A. Manual Verification (Primary Method):**
1.  Start the development server using the command in Step 3 above.
2.  Open the application in a browser at `http://localhost:8899`.
3.  Log in and manually navigate through the application to test the feature you have changed.
4.  Observe the server logs for errors: `tail -f /tmp/lecapp_server.log`.

**B. Running Existing Automated Tests:**
If you need to run the existing test suite to check for regressions:
1.  Make sure your virtual environment is active.
2.  Run the command: `pytest -q`

---

### **5. Common Tasks**

*   **Reset Database (Local Development Only):**
    If you need to clear all data and start fresh, run this script:
    ```bash
    python scripts/reset_db.py
    ```

*   **Create an Admin User:**
    If no admin user exists, run this interactive script:
    ```bash
    python scripts/create_admin.py
    ```

---

### **6. Quick Reference**

| Item | Value / Command |
| :--- | :--- |
| **Application URL** | `http://localhost:8899` |
| **Server Log File** | `/tmp/lecapp_server.log` |
| **Local Database Path** | `./instance/transcriptions.db` (controlled by `.env`) |
| **Local Uploads Path**| `./uploads` (controlled by `.env`) |
| **Enable Inquire Mode**| In `.env`, set `ENABLE_INQUIRE_MODE=true` |
| **Enable Auto-Processing**| In `.env`, set `ENABLE_AUTO_PROCESSING=true` |