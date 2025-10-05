# Quick Start Guide

Get LecApp up and running in just a few minutes using the pre-built Docker image! This guide will walk you through the fastest way to deploy LecApp with either OpenAI Whisper API or a [custom ASR endpoint](features.md#speaker-diarization).

> **Note:** If you want to use the ASR endpoint option for speaker diarization features, you'll need to run an additional Docker container (`onerahmet/openai-whisper-asr-webservice`). See [Running ASR Service for Speaker Diarization](getting-started/installation.md#running-asr-service-for-speaker-diarization) for detailed setup instructions.

## Prerequisites

Before you begin, make sure you have Docker and Docker Compose installed on your system. You'll also need an API key for either OpenAI or OpenRouter (or a compatible service), at least 2GB of available RAM, and about 10GB of available disk space for storing recordings and transcriptions.

## Step 1: Create Project Directory

First, create a directory for your LecApp installation and navigate into it:

```bash
mkdir lecapp
cd lecapp
```

## Step 2: Download Configuration Files

Download the Docker Compose configuration and choose the appropriate environment template based on your transcription service:

```bash
# Download docker compose example
wget https://raw.githubusercontent.com/baveshm/LecApp/master/config/docker-compose.example.yml -O docker-compose.yml
```

Now download the environment configuration template. You have two options depending on which transcription service you want to use.

For standard OpenAI Whisper API (recommended for most users):
```bash
wget https://raw.githubusercontent.com/baveshm/LecApp/master/config/env.whisper.example -O .env
```

Or for a custom ASR endpoint with speaker diarization (requires additional ASR container - see note below):
```bash
wget https://raw.githubusercontent.com/baveshm/LecApp/master/config/env.asr.example -O .env
```

> **Important:** The ASR endpoint option requires running an additional Docker container (`onerahmet/openai-whisper-asr-webservice`) alongside LecApp. For complete setup instructions including docker-compose configurations for both containers, see [Running ASR Service for Speaker Diarization](getting-started/installation.md#running-asr-service-for-speaker-diarization).

## Step 3: Configure Your Transcription Service

Open the `.env` file in your preferred text editor and configure it based on your chosen service.

### Option A: OpenAI Whisper Configuration

If you're using OpenAI Whisper, you'll need to set up both the transcription service and the text generation model. The text generation model is used for creating summaries, generating titles, and powering the chat features.

Edit your `.env` file and update these key variables:

```bash
# For text generation (summaries, chat, titles)
TEXT_MODEL_BASE_URL=https://openrouter.ai/api/v1
TEXT_MODEL_API_KEY=your_openrouter_api_key_here
TEXT_MODEL_NAME=openai/gpt-4o-mini

# For transcription
TRANSCRIPTION_BASE_URL=https://api.openai.com/v1
TRANSCRIPTION_API_KEY=your_openai_api_key_here
WHISPER_MODEL=whisper-1
```

The text model can use OpenRouter for access to various AI models, or you can point it directly to OpenAI by using the same base URL and API key as your transcription service. OpenRouter provides access to multiple models including GPT-4, Claude, and others, which can be more cost-effective for text generation tasks.

### Option B: Custom ASR Endpoint Configuration

> **Prerequisites:** This option requires running an additional ASR service container (`onerahmet/openai-whisper-asr-webservice`). You can either:
>
> - Run both containers in the same Docker Compose stack (recommended) - see [complete setup guide](getting-started/installation.md#running-asr-service-for-speaker-diarization)
> - Run the ASR service separately on the same or different machine
> - Use an existing ASR service if you already have one deployed

If you're using a custom ASR service like WhisperX or a self-hosted Whisper server, configure these variables:

```bash
# For text generation (summaries, chat, titles)
TEXT_MODEL_BASE_URL=https://openrouter.ai/api/v1
TEXT_MODEL_API_KEY=your_openrouter_api_key_here
TEXT_MODEL_NAME=openai/gpt-4o-mini

# Enable ASR endpoint
USE_ASR_ENDPOINT=true

# ASR service URL (use container name if in same docker compose)
ASR_BASE_URL=http://whisper-asr:9000
```

When using an ASR endpoint, [speaker diarization](features.md#speaker-diarization) is automatically enabled, allowing LecApp to identify different speakers in your recordings. After transcription, you'll need to [identify speakers](user-guide/transcripts.md#speaker-identification) to build your speaker library. The ASR_BASE_URL should point to your ASR service. If you're running the ASR service in the same Docker Compose stack, use the container name and internal port (like `http://whisper-asr:9000`). For external services, use the full URL with the appropriate IP address or domain name.

## Step 4: Configure Admin Account

LecApp automatically creates an admin user on first startup. Configure these credentials in your `.env` file before launching:

```bash
ADMIN_USERNAME=admin
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=changeme
```

Make sure to change these values to something secure, especially the password. This admin account will be created automatically when you first start LecApp, and you'll use these credentials to log in. The first user created through this method becomes the system administrator with full access to all features including user management and system settings.

## Step 5: Launch LecApp

With your configuration complete, start LecApp using Docker Compose:

```bash
docker compose up -d
```

The first launch will take a few minutes as Docker downloads the pre-built image (about 3GB) and initializes the database. You can monitor the startup process by viewing the logs:

```bash
docker compose logs -f app
```

Look for a message indicating that the Flask application is running and ready to accept connections. Press Ctrl+C to exit the log view (this won't stop the container).

## Step 6: Access LecApp

Once the container is running, open your web browser and navigate to:

```
http://localhost:8899
```

Log in using the admin credentials you configured in Step 4. You should now see the LecApp dashboard, ready for your first recording.

## Your First Recording

After logging in, you can immediately start using LecApp. Click the "New Recording" button in the top navigation to either upload an existing audio file or start a [live recording](user-guide/recording.md). For detailed instructions, see the [recording guide](user-guide/recording.md). For uploads, LecApp supports [common audio formats](faq.md#what-audio-formats-does-lecapp-support) like MP3, M4A, WAV, and more, with files up to 500MB by default. You can adjust this limit in [system settings](admin-guide/system-settings.md). For live recording, you can capture from your microphone, system audio, or both simultaneously.

## Optional Features

### Enable Inquire Mode

[Inquire Mode](user-guide/inquire-mode.md) allows you to search across all your recordings using natural language questions. Learn more about [semantic search capabilities](features.md#semantic-search-inquire-mode) in the features guide. To enable it, set this in your `.env` file:

```bash
ENABLE_INQUIRE_MODE=true
```

Then restart the container with `docker compose restart` for the change to take effect.

### Enable User Registration

By default, only the admin can create new users. Learn more about [user management](admin-guide/user-management.md) in the admin guide. To allow self-registration, set:

```bash
ALLOW_REGISTRATION=true
```

### Configure Your Timezone

Set your local timezone for accurate timestamp display:

```bash
TIMEZONE="America/New_York"
```

Use any valid timezone from the TZ database like "Europe/London", "Asia/Tokyo", or "UTC".

## Stopping and Starting LecApp

To stop LecApp while preserving all your data:

```bash
docker compose down
```

To start it again:

```bash
docker compose up -d
```

Your recordings, transcriptions, and settings are preserved in the `./uploads` and `./instance` directories on your host system.

## Troubleshooting

If LecApp doesn't start properly, check the logs for error messages using `docker compose logs app`. For more detailed help, see the [Troubleshooting Guide](troubleshooting.md), particularly the [installation issues](troubleshooting.md#installation-and-setup-issues) section. Common issues include incorrect API keys, which will show authentication errors in the logs, or port conflicts if another service is using port 8899. You can change the port by editing the `docker-compose.yml` file and modifying the ports section.

If transcription fails, verify your API keys are correct and you have sufficient credits with your chosen service. The logs will show detailed error messages that can help identify the issue.

---

Next: [Installation Guide](getting-started/installation.md) for production deployments and advanced configuration