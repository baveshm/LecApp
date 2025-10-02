# LacApp — Start-up Guide for Agents

This guide shows the quickest, reliable ways to start the Speakr app for demos, testing, or local development.

- Default URL: http://localhost:8899
- Default database: SQLite
- Recommended method: Docker Compose

## 1) Start with Docker Compose (recommended)

Works the same on macOS, Linux, and Windows (with Docker Desktop). Minimal setup and fully reproducible.

### Steps

1. From the repo root, copy the example compose and env file:

```bash
cp config/docker-compose.example.yml docker-compose.yml
# Choose ONE env template:
cp config/env.whisper.example .env   # Standard Whisper API (OpenAI-compatible)
# or
cp config/env.asr.example .env       # ASR endpoint with speaker diarization
```

2. Edit .env and set your keys and settings:

- TEXT_MODEL_BASE_URL, TEXT_MODEL_API_KEY, TEXT_MODEL_NAME
- For Whisper API: TRANSCRIPTION_BASE_URL, TRANSCRIPTION_API_KEY, WHISPER_MODEL
- For ASR endpoint: USE_ASR_ENDPOINT=true, ASR_BASE_URL=http://whisper-asr:9000 (if using a companion ASR container)
- Admin bootstrap on first run:
  ADMIN_USERNAME, ADMIN_EMAIL, ADMIN_PASSWORD

3. Create data directories (bind mounts used by docker-compose.yml):

```bash
mkdir -p uploads instance
```

4. Start the app:

```bash
docker compose up -d
```

5. Open http://localhost:8899 and log in with the admin credentials set in .env.

Notes
- If using ASR with diarization, run the ASR service container as well (see docs/getting-started/installation.md for compose examples). On macOS, use the CPU image of the ASR service.
- View logs: `docker compose logs -f app`
- Stop: `docker compose down`

## 2) Run from source (local Python)

Useful for quick edits and debugging without containers. Requires Python 3.11+ and ffmpeg installed on your system.

### Steps

1. Set up a virtual environment and install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Copy an env template and adjust for local paths:

```bash
cp config/env.whisper.example .env   # or env.asr.example
```

Edit .env and (recommended) override these two for local dev so files live in the project folder:

```bash
SQLALCHEMY_DATABASE_URI=sqlite:///./instance/transcriptions.db
UPLOAD_FOLDER=./uploads
```

Then set your API keys and admin bootstrap values as described above.

3. Create data directories:

```bash
mkdir -p uploads instance
```

4. Create an admin user (interactive):

```bash
python scripts/create_admin.py
```

5. Start the server:

- Production-like:

```bash
gunicorn --workers 3 --bind 0.0.0.0:8899 --timeout 600 src.app:app
```

- Or lightweight dev server (auto-reload):

```bash
export FLASK_APP=src/app.py
flask run --host 0.0.0.0 --port 8899
```

Then open http://localhost:8899.

## Optional features

- Inquire Mode (semantic search across all recordings): set `ENABLE_INQUIRE_MODE=true` in .env. The Docker image already contains required packages; for local runs, requirements.txt includes sentence-transformers and scikit-learn.
- Automated file processing (“black hole”): set `ENABLE_AUTO_PROCESSING=true` and adjust `AUTO_PROCESS_*` variables. Ensure the watch directory exists and is writable.

## Troubleshooting

- Port already in use: change the host port mapping in docker-compose.yml (e.g., `"8080:8899"`) or stop the conflicting service.
- Admin login fails: verify .env values and rerun admin creation (Docker auto-creates admin on first run; local uses `scripts/create_admin.py`).
- Missing ASR results/diarization: ensure `USE_ASR_ENDPOINT=true` and `ASR_BASE_URL` points to a reachable ASR service; check its logs.
- Large files with Whisper API: enable chunking in .env (`ENABLE_CHUNKING=true`, set `CHUNK_LIMIT` and `CHUNK_OVERLAP_SECONDS`). Chunking is not used with ASR endpoints.

## Quick reference

- URL: http://localhost:8899
- Docker logs: `docker compose logs -f app`
- Local logs: printed to terminal; adjust `LOG_LEVEL` in .env
- Data paths (Docker): `./uploads` and `./instance` on the host
- Data paths (Local): as set by `UPLOAD_FOLDER` and `SQLALCHEMY_DATABASE_URI`

---

For full docs, see README.md and docs/getting-started/installation.md.
