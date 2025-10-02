# Speakr — Agent Runbook (Codex/Copilot)

This file is for an automated coding agent to reliably run the web server and make safe edits.

- Default URL: http://localhost:8899
- Default DB: SQLite (file lives under `./instance/` locally or `/data/instance/` in Docker)
- Preferred start method: Docker Compose (fully reproducible)

## How to run the web server

### Option A — Docker Compose (recommended)

Prereqs: Docker Desktop (macOS/Windows) or Docker Engine (Linux).

1) Prepare config at repo root

```bash
cp config/docker-compose.example.yml docker-compose.yml
# Choose ONE env template
cp config/env.whisper.example .env   # Whisper API (OpenAI-compatible)
# or
cp config/env.asr.example .env       # ASR endpoint + speaker diarization
```

2) Edit `.env` and set keys

- Text model: `TEXT_MODEL_BASE_URL`, `TEXT_MODEL_API_KEY`, `TEXT_MODEL_NAME`
- Whisper API flow: `TRANSCRIPTION_BASE_URL`, `TRANSCRIPTION_API_KEY`, `WHISPER_MODEL`
- ASR flow: `USE_ASR_ENDPOINT=true`, `ASR_BASE_URL=http://whisper-asr:9000`
- Admin bootstrap (auto-created on first run): `ADMIN_USERNAME`, `ADMIN_EMAIL`, `ADMIN_PASSWORD`

3) Create host data dirs

```bash
mkdir -p uploads instance
```

4) Start server

```bash
docker compose up -d
```

5) Health check

- Open http://localhost:8899/login and confirm HTTP 200 and the login form renders
- Logs: `docker compose logs -f app`
- Stop: `docker compose down`

Notes
- For diarization, run the ASR webservice container too (see `docs/getting-started/installation.md`). On macOS use CPU image.

### Option B — Run locally (Python)

Prereqs: Python 3.11+, ffmpeg installed (macOS: `brew install ffmpeg`).

1) Create venv and install deps

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Env and local paths

```bash
cp config/env.whisper.example .env   # or config/env.asr.example
```

Required local overrides (keep data inside project directory):

```bash
SQLALCHEMY_DATABASE_URI=sqlite:///./instance/transcriptions.db
UPLOAD_FOLDER=./uploads
```

3) Create data dirs

```bash
mkdir -p uploads instance
```

4) Create admin user (interactive)

```bash
python scripts/create_admin.py
```

5) Start server

- Production-like:

```bash
gunicorn --workers 3 --bind 0.0.0.0:8899 --timeout 600 src.app:app
```

- Dev server with auto-reload:

```bash
export FLASK_APP=src/app.py
flask run --host 0.0.0.0 --port 8899
```

6) Health check

- Open http://localhost:8899/login and verify HTTP 200 with login page
- Server logs appear in the same terminal

## Editing workflow for the agent

Where to make changes
- Backend routes, logic, and config: `src/app.py`
- Templates (HTML/Jinja): `templates/` (e.g., `templates/index.html`, `templates/login.html`)
- Frontend assets: `static/js/` and `static/css/`
- Background helpers: `src/audio_chunking.py`, `src/file_monitor.py`
- Scripts and utilities: `scripts/`
- Tests: `tests/`

Safe edit-and-verify loop (local dev)
1) Ensure venv active and deps installed (see run locally steps)
2) Run dev server: `flask run --host 0.0.0.0 --port 8899`
3) Make code changes, save, and verify changes hot-reload in the browser
4) Run tests (optional, see below)

Safe edit-and-verify loop (Docker)
1) Edit files on host (volumes are bind-mounted in example compose)
2) If using gunicorn in container, it won’t auto-reload — restart container after edits:

```bash
docker compose restart app
```

Testing
- If pytest isn’t available, install it: `pip install pytest`
- Run all tests: `pytest -q`
- Relevant suites live under `tests/` (e.g., inquire mode, JSON fix/preprocessing)

Adding/Updating dependencies
1) Add pinned package to `requirements.txt`
2) Reinstall: `pip install -r requirements.txt`
3) For Docker, rebuild image only if system deps change; otherwise container pulls prebuilt image

Common tasks
- Reset DB (local dev only): use `scripts/reset_db.py` to clear state
- Recreate admin: rerun `scripts/create_admin.py` (local) or set `ADMIN_*` in `.env` (Docker) and restart
- Change port: update `docker-compose.yml` mapping (e.g., `"8080:8899"`) or change `--port` in dev server

## Options and flags you may need

- Inquire Mode (semantic search): set `ENABLE_INQUIRE_MODE=true` in `.env`
- Automated file processing: set `ENABLE_AUTO_PROCESSING=true` and configure `AUTO_PROCESS_*`
- Chunking for large files (Whisper API flow only): `ENABLE_CHUNKING=true`, set `CHUNK_LIMIT`, `CHUNK_OVERLAP_SECONDS`

## Troubleshooting quick refs

- Port in use: adjust compose port mapping or stop the other process
- Admin login fails: confirm `.env` values and admin creation step
- ASR features missing: ensure `USE_ASR_ENDPOINT=true` and `ASR_BASE_URL` points to reachable service
- ffmpeg missing errors: install ffmpeg (macOS: `brew install ffmpeg`)

## Quick reference

- App URL: http://localhost:8899
- Docker logs: `docker compose logs -f app`
- Local logs: printed to terminal; adjust `LOG_LEVEL` in `.env`
- Data paths (Docker): host `./uploads` and `./instance` map into container
- Data paths (Local): controlled by `UPLOAD_FOLDER` and `SQLALCHEMY_DATABASE_URI`

—

For deeper details, see `README.md` and `docs/getting-started/installation.md`.
