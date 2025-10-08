"""
Transcription Service for LecApp.

This module handles all audio transcription operations including:
- Audio extraction from video containers
- ASR endpoint transcription
- Whisper API transcription
- Audio chunking for large files
"""

import os
import json
import subprocess
import mimetypes
import time
import logging
from datetime import datetime
from openai import OpenAI
import httpx
from dotenv import load_dotenv

# Load environment variables from project root
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
load_dotenv(_env_path)

from src.extensions import db
from src.models import Recording, SystemSetting
from src.services.llm_service import generate_title_task, generate_summary_only_task
from src.audio_chunking import AudioChunkingService, ChunkProcessingError, ChunkingNotSupportedError

# Get logger
logger = logging.getLogger(__name__)

# --- Transcription API Configuration ---
transcription_api_key = os.environ.get("TRANSCRIPTION_API_KEY", "")
transcription_base_url = os.environ.get("TRANSCRIPTION_BASE_URL", "")
if transcription_base_url:
    transcription_base_url = transcription_base_url.split('#')[0].strip()

# --- ASR Endpoint Configuration ---
USE_ASR_ENDPOINT = os.environ.get('USE_ASR_ENDPOINT', 'false').lower() == 'true'
ASR_BASE_URL = os.environ.get('ASR_BASE_URL')
if ASR_BASE_URL:
    ASR_BASE_URL = ASR_BASE_URL.split('#')[0].strip()

# When using ASR endpoint, automatically enable diarization and set sensible defaults
if USE_ASR_ENDPOINT:
    ASR_DIARIZE = os.environ.get('ASR_DIARIZE', 'true').lower() == 'true'
    ASR_MIN_SPEAKERS = os.environ.get('ASR_MIN_SPEAKERS')
    ASR_MAX_SPEAKERS = os.environ.get('ASR_MAX_SPEAKERS')
else:
    ASR_DIARIZE = False
    ASR_MIN_SPEAKERS = None
    ASR_MAX_SPEAKERS = None

# --- Audio Chunking Configuration ---
ENABLE_CHUNKING = os.environ.get('ENABLE_CHUNKING', 'true').lower() == 'true'
CHUNK_SIZE_MB = int(os.environ.get('CHUNK_SIZE_MB', '20'))
CHUNK_OVERLAP_SECONDS = int(os.environ.get('CHUNK_OVERLAP_SECONDS', '3'))

# Initialize chunking service
chunking_service = AudioChunkingService(
    max_chunk_size_mb=CHUNK_SIZE_MB,
    overlap_seconds=CHUNK_OVERLAP_SECONDS
) if ENABLE_CHUNKING else None

# HTTP client for transcription API
http_client_no_proxy = httpx.Client(verify=True)


# --- Audio Processing Functions ---

def extract_audio_from_video(video_filepath, output_format='mp3', cleanup_original=True):
    """Extract audio from video containers using FFmpeg.
    
    Uses MP3 codec for optimal compatibility and predictable file sizes.
    64kbps MP3 provides good speech quality at ~480KB per minute.
    """
    try:
        # Generate output filename with audio extension
        base_filepath, file_ext = os.path.splitext(video_filepath)
        temp_audio_filepath = f"{base_filepath}_audio_temp.{output_format}"
        final_audio_filepath = f"{base_filepath}_audio.{output_format}"
        
        logger.info(f"Extracting audio from video: {video_filepath} -> {temp_audio_filepath}")
        
        # Extract audio using FFmpeg - using high-quality MP3 for better transcription
        subprocess.run([
            'ffmpeg', '-i', video_filepath, '-y',
            '-vn',  # No video
            '-codec:a', 'libmp3lame',  # Use LAME MP3 encoder explicitly
            '-b:a', '128k',  # 128kbps bitrate for high quality
            '-ar', '44100',  # 44.1kHz sample rate for better quality
            '-ac', '1',  # Mono (sufficient for speech, reduces file size)
            '-compression_level', '2',  # Better compression
            temp_audio_filepath
        ], check=True, capture_output=True, text=True)
        
        logger.info(f"Successfully extracted audio to {temp_audio_filepath}")
        
        # Optionally preserve temp file for debugging (set PRESERVE_TEMP_AUDIO=true in env)
        if os.getenv('PRESERVE_TEMP_AUDIO', 'false').lower() == 'true':
            import shutil
            shutil.copy2(temp_audio_filepath, temp_audio_filepath.replace('_temp', '_debug'))
            logger.info(f"Debug: Preserved temp audio file as {temp_audio_filepath.replace('_temp', '_debug')}")
        
        # Rename temp file to final filename
        os.rename(temp_audio_filepath, final_audio_filepath)
        
        # Clean up original video file if requested
        if cleanup_original:
            try:
                os.remove(video_filepath)
                logger.info(f"Cleaned up original video file: {video_filepath}")
            except Exception as e:
                logger.warning(f"Failed to clean up original video file {video_filepath}: {str(e)}")
        
        return final_audio_filepath, f'audio/{output_format}'
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg audio extraction failed for {video_filepath}: {e.stderr}")
        raise Exception(f"Audio extraction failed: {e.stderr}")
    except FileNotFoundError:
        logger.error("FFmpeg command not found. Please ensure FFmpeg is installed and in the system's PATH.")
        raise Exception("Audio conversion tool (FFmpeg) not found on server.")
    except Exception as e:
        logger.error(f"Error extracting audio from {video_filepath}: {str(e)}")
        raise


# --- ASR Transcription Functions ---

def transcribe_audio_asr(app_context, recording_id, filepath, original_filename, start_time, mime_type=None, language=None, diarize=False, min_speakers=None, max_speakers=None, tag_id=None):
    """Transcribes audio using the ASR webservice."""
    with app_context:
        recording = db.session.get(Recording, recording_id)
        if not recording:
            logger.error(f"Error: Recording {recording_id} not found for ASR transcription.")
            return

        try:
            logger.info(f"Starting ASR transcription for recording {recording_id}...")
            recording.status = 'PROCESSING'
            db.session.commit()

            # Check if we need to extract audio from video container
            actual_filepath = filepath
            actual_content_type = mime_type or mimetypes.guess_type(original_filename)[0] or 'application/octet-stream'
            actual_filename = original_filename

            # List of video MIME types that need audio extraction
            video_mime_types = [
                'video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm',
                'video/avi', 'video/x-ms-wmv', 'video/3gpp'
            ]
            
            # Check if file is a video container by MIME type or extension
            is_video = (
                actual_content_type.startswith('video/') or 
                actual_content_type in video_mime_types or
                original_filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv', '.3gp'))
            )
            
            if is_video:
                logger.info(f"Video container detected ({actual_content_type}), extracting audio...")
                try:
                    # Extract audio from video
                    audio_filepath, audio_mime_type = extract_audio_from_video(filepath, 'mp3')
                    
                    # Update paths and MIME type for ASR processing
                    actual_filepath = audio_filepath
                    actual_content_type = audio_mime_type
                    actual_filename = os.path.basename(audio_filepath)
                    
                    # Update recording with extracted audio path and new MIME type
                    recording.audio_path = audio_filepath
                    recording.mime_type = audio_mime_type
                    db.session.commit()
                    
                    logger.info(f"Audio extracted successfully: {audio_filepath}")
                except Exception as e:
                    logger.error(f"Failed to extract audio from video: {str(e)}")
                    recording.status = 'FAILED'
                    recording.error_msg = f"Audio extraction failed: {str(e)}"
                    db.session.commit()
                    return

            # Keep track of whether we've already tried WAV conversion
            wav_conversion_attempted = False
            wav_converted_filepath = None
            
            # Retry loop for handling 500 errors with WAV conversion
            max_attempts = 2
            for attempt in range(max_attempts):
                try:
                    # Use converted MP3 if available from previous attempt
                    current_filepath = wav_converted_filepath if wav_converted_filepath else actual_filepath
                    current_content_type = 'audio/mpeg' if wav_converted_filepath else actual_content_type
                    current_filename = os.path.basename(current_filepath)
                    
                    with open(current_filepath, 'rb') as audio_file:
                        url = f"{ASR_BASE_URL}/asr"
                        params = {
                            'encode': True,
                            'task': 'transcribe',
                            'output': 'json'
                        }
                        if language:
                            params['language'] = language
                        if diarize:
                            params['diarize'] = diarize
                        if min_speakers:
                            params['min_speakers'] = min_speakers
                        if max_speakers:
                            params['max_speakers'] = max_speakers

                        content_type = current_content_type
                        logger.info(f"Using MIME type {content_type} for ASR upload.")
                        files = {'audio_file': (current_filename, audio_file, content_type)}
                        
                        with httpx.Client() as client:
                            # Get configurable ASR timeout from database (default 30 minutes)
                            asr_timeout_seconds = SystemSetting.get_setting('asr_timeout_seconds', 1800)
                            timeout = httpx.Timeout(None, connect=30.0, read=float(asr_timeout_seconds), write=30.0, pool=30.0)
                            logger.info(f"Sending ASR request to {url} with params: {params} (timeout: {asr_timeout_seconds}s)")
                            response = client.post(url, params=params, files=files, timeout=timeout)
                            logger.info(f"ASR request completed with status: {response.status_code}")
                            response.raise_for_status()
                            
                            # Parse the JSON response from ASR (moved here so it's accessible)
                            asr_response_data = response.json()
                    
                    # If we reach here, the request was successful
                    break
                    
                except httpx.HTTPStatusError as e:
                    # Check if it's a 500 error and we haven't tried WAV conversion yet
                    if e.response.status_code == 500 and attempt == 0 and not wav_conversion_attempted:
                        logger.warning(f"ASR returned 500 error for recording {recording_id}, attempting high-quality MP3 conversion and retry...")
                        
                        # Convert to high-quality MP3 for better compatibility
                        filename_lower = actual_filename.lower()
                        if not filename_lower.endswith('.mp3'):
                            try:
                                base_filepath, file_ext = os.path.splitext(actual_filepath)
                                temp_mp3_filepath = f"{base_filepath}_temp.mp3"
                                
                                logger.info(f"Converting {actual_filename} to high-quality MP3 format for retry...")
                                subprocess.run(
                                    ['ffmpeg', '-i', actual_filepath, '-y', '-acodec', 'libmp3lame', '-b:a', '128k', '-ar', '44100', temp_mp3_filepath],
                                    check=True, capture_output=True, text=True
                                )
                                logger.info(f"Successfully converted {actual_filepath} to {temp_mp3_filepath}")
                                
                                wav_converted_filepath = temp_mp3_filepath  # Keep variable name for compatibility
                                wav_conversion_attempted = True
                                # Continue to next iteration to retry with WAV
                                continue
                            except subprocess.CalledProcessError as conv_error:
                                logger.error(f"Failed to convert to WAV: {conv_error}")
                                # Re-raise the original HTTP error if conversion fails
                                raise e
                        else:
                            # Already a WAV file, can't convert further
                            logger.error(f"File is already WAV but still getting 500 error")
                            raise e
                    else:
                        # Not a 500 error or already tried conversion, propagate the error
                        raise e
            
            # DEBUG: Preserve converted file for quality checking
            if wav_converted_filepath and os.path.exists(wav_converted_filepath):
                try:
                    # Get file size and basic info for debugging
                    converted_size = os.path.getsize(wav_converted_filepath)
                    converted_size_mb = converted_size / (1024 * 1024)
                    
                    # Create a debug copy in a known location
                    from flask import current_app
                    debug_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'debug_converted')
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    # Copy the converted file with a timestamp (MP3 now)
                    from shutil import copy2
                    file_ext = os.path.splitext(wav_converted_filepath)[1] or '.mp3'
                    debug_filename = f"debug_{recording_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}{file_ext}"
                    debug_filepath = os.path.join(debug_dir, debug_filename)
                    copy2(wav_converted_filepath, debug_filepath)
                    
                    logger.info(f"DEBUG: Converted file preserved at: {debug_filepath}")
                    logger.info(f"DEBUG: Converted file size: {converted_size_mb:.2f} MB ({converted_size} bytes)")
                    logger.info(f"DEBUG: Original file: {actual_filename}")
                    logger.info(f"DEBUG: Recording ID: {recording_id}")
                    logger.info(f"DEBUG: You can download this file from the container at: {debug_filepath}")
                    
                except Exception as debug_error:
                    logger.warning(f"DEBUG: Failed to preserve converted file: {debug_error}")
            
            # Clean up the original temporary converted file (but keep debug copy)
            try:
                if wav_converted_filepath and os.path.exists(wav_converted_filepath):
                    os.remove(wav_converted_filepath)
                    logger.info(f"Cleaned up original temporary converted file: {wav_converted_filepath}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary converted file: {cleanup_error}")
            
            # Debug logging for ASR response
            logger.info(f"ASR response keys: {list(asr_response_data.keys())}")
            
            # Log the complete raw JSON response (truncated for readability)
            import json as json_module
            raw_json_str = json_module.dumps(asr_response_data, indent=2)
            if len(raw_json_str) > 5000:
                logger.info(f"Raw ASR response (first 5000 chars): {raw_json_str[:5000]}...")
            else:
                logger.info(f"Raw ASR response: {raw_json_str}")
            
            if 'segments' in asr_response_data:
                logger.info(f"Number of segments: {len(asr_response_data['segments'])}")
                
                # Collect all unique speakers from the response
                all_speakers = set()
                segments_with_speakers = 0
                segments_without_speakers = 0
                
                for segment in asr_response_data['segments']:
                    if 'speaker' in segment and segment['speaker'] is not None:
                        all_speakers.add(segment['speaker'])
                        segments_with_speakers += 1
                    else:
                        segments_without_speakers += 1
                
                logger.info(f"Unique speakers found in raw response: {sorted(list(all_speakers))}")
                logger.info(f"Segments with speakers: {segments_with_speakers}, without speakers: {segments_without_speakers}")
                
                # Log first few segments for debugging
                for i, segment in enumerate(asr_response_data['segments'][:5]):
                    segment_keys = list(segment.keys())
                    logger.info(f"Segment {i} keys: {segment_keys}")
                    logger.info(f"Segment {i}: speaker='{segment.get('speaker')}', text='{segment.get('text', '')[:50]}...'")
            
            # Simplify the JSON data
            simplified_segments = []
            if 'segments' in asr_response_data and isinstance(asr_response_data['segments'], list):
                last_known_speaker = None
                
                for i, segment in enumerate(asr_response_data['segments']):
                    speaker = segment.get('speaker')
                    text = segment.get('text', '').strip()
                    
                    # If segment doesn't have a speaker, use the previous segment's speaker
                    if speaker is None:
                        if last_known_speaker is not None:
                            speaker = last_known_speaker
                            logger.info(f"Assigned speaker '{speaker}' to segment {i} from previous segment")
                        else:
                            speaker = 'UNKNOWN_SPEAKER'
                            logger.warning(f"No previous speaker available for segment {i}, using UNKNOWN_SPEAKER")
                    else:
                        # Update the last known speaker when we have a valid one
                        last_known_speaker = speaker
                    
                    simplified_segments.append({
                        'speaker': speaker,
                        'sentence': text,
                        'start_time': segment.get('start'),
                        'end_time': segment.get('end')
                    })
            
            # Log final simplified segments count
            logger.info(f"Created {len(simplified_segments)} simplified segments")
            null_speaker_count = sum(1 for seg in simplified_segments if seg['speaker'] is None)
            if null_speaker_count > 0:
                logger.warning(f"Found {null_speaker_count} segments with null speakers in final output")
            
            # Store the simplified JSON as a string
            recording.transcription = json.dumps(simplified_segments)
            
            # Commit the transcription data
            db.session.commit()
            logger.info(f"ASR transcription completed for recording {recording_id}.")
            
            # Generate title immediately
            generate_title_task(app_context, recording_id)
            
            # Always auto-generate summary for all recordings
            logger.info(f"Auto-generating summary for recording {recording_id}")
            generate_summary_only_task(app_context, recording_id)

        except Exception as e:
            db.session.rollback()
            
            # Handle timeout errors specifically
            error_msg = str(e)
            if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
                asr_timeout = SystemSetting.get_setting('asr_timeout_seconds', 1800)
                logger.error(f"ASR processing TIMED OUT for recording {recording_id} after {asr_timeout} seconds. Consider increasing 'asr_timeout_seconds' in Admin Dashboard > System Settings.")
                user_error_msg = f"ASR processing timed out after {asr_timeout} seconds. The file may be too long for the current timeout setting."
            else:
                # For non-timeout errors, include more detail
                logger.error(f"ASR processing FAILED for recording {recording_id}: {error_msg}")
                user_error_msg = f"ASR processing failed: {error_msg}"
            
            recording = db.session.get(Recording, recording_id)
            if recording:
                recording.status = 'FAILED'
                recording.transcription = user_error_msg
                db.session.commit()


# --- Whisper API Transcription Functions ---

def transcribe_audio_task(app_context, recording_id, filepath, filename_for_asr, start_time, language=None, min_speakers=None, max_speakers=None, tag_id=None):
    """Runs the transcription and summarization in a background thread.
    
    Args:
        app_context: Flask app context
        recording_id: ID of the recording to process
        filepath: Path to the audio file
        filename_for_asr: Filename to use for ASR
        start_time: Processing start time
        language: Optional language code override (from upload form)
        min_speakers: Optional minimum speakers override (from upload form)
        max_speakers: Optional maximum speakers override (from upload form)
        tag_id: Optional tag ID to apply custom prompt from
    """
    if USE_ASR_ENDPOINT:
        with app_context:
            recording = db.session.get(Recording, recording_id)
            # Environment variable ASR_DIARIZE overrides user setting
            if 'ASR_DIARIZE' in os.environ:
                diarize_setting = ASR_DIARIZE
            elif USE_ASR_ENDPOINT:
                # When using ASR endpoint, use the configured ASR_DIARIZE value
                diarize_setting = ASR_DIARIZE
            else:
                diarize_setting = recording.owner.diarize if recording.owner else False
            
            # Use language from upload form if provided, otherwise use user's default
            if language:
                user_transcription_language = language
            else:
                user_transcription_language = recording.owner.transcription_language if recording.owner else None
        # Use min/max speakers from upload form (already processed with precedence hierarchy)
        # If None, ASR will auto-detect the number of speakers
        final_min_speakers = min_speakers
        final_max_speakers = max_speakers
        
        transcribe_audio_asr(app_context, recording_id, filepath, filename_for_asr, start_time, 
                           mime_type=recording.mime_type, 
                           language=user_transcription_language, 
                           diarize=diarize_setting,
                           min_speakers=final_min_speakers,
                           max_speakers=final_max_speakers,
                           tag_id=tag_id)
        
        # After ASR task completes, calculate processing time
        with app_context:
            recording = db.session.get(Recording, recording_id)
            if recording.status in ['COMPLETED', 'FAILED']:
                end_time = datetime.utcnow()
                recording.processing_time_seconds = (end_time - start_time).total_seconds()
                db.session.commit()
        return

    with app_context: # Need app context for db operations in thread
        recording = db.session.get(Recording, recording_id)
        if not recording:
            logger.error(f"Error: Recording {recording_id} not found for transcription.")
            return

        try:
            logger.info(f"Starting transcription for recording {recording_id} ({filename_for_asr})...")
            recording.status = 'PROCESSING'
            db.session.commit()

            # Check if chunking is needed for large files
            needs_chunking = (chunking_service and 
                            ENABLE_CHUNKING and 
                            chunking_service.needs_chunking(filepath, USE_ASR_ENDPOINT))
            
            if needs_chunking:
                logger.info(f"File {filepath} is large ({os.path.getsize(filepath)/1024/1024:.1f}MB), using chunking for transcription")
                transcription_text = transcribe_with_chunking(app_context, recording_id, filepath, filename_for_asr)
            else:
                # --- Standard transcription for smaller files ---
                transcription_text = transcribe_single_file(filepath, recording)
            
            recording.transcription = transcription_text
            logger.info(f"Transcription completed for recording {recording_id}. Text length: {len(recording.transcription)}")
            
            # Generate title immediately
            generate_title_task(app_context, recording_id)
            
            # Always auto-generate summary for all recordings
            logger.info(f"Auto-generating summary for recording {recording_id}")
            generate_summary_only_task(app_context, recording_id)

        except Exception as e:
            db.session.rollback() # Rollback if any step failed critically
            logger.error(f"Processing FAILED for recording {recording_id}: {str(e)}", exc_info=True)
            # Retrieve recording again in case session was rolled back
            recording = db.session.get(Recording, recording_id)
            if recording:
                 # Ensure status reflects failure even after rollback/retrieve attempt
                if recording.status not in ['COMPLETED', 'FAILED']: # Avoid overwriting final state
                    recording.status = 'FAILED'
                if not recording.transcription: # If transcription itself failed
                     recording.transcription = f"Processing failed: {str(e)}"
                # Add error note to summary if appropriate stage was reached
                if recording.status == 'SUMMARIZING' and not recording.summary:
                     recording.summary = f"[Processing failed during summarization: {str(e)}]"
                
                end_time = datetime.utcnow()
                recording.processing_time_seconds = (end_time - start_time).total_seconds()
                db.session.commit()


def transcribe_single_file(filepath, recording):
    """Transcribe a single audio file using OpenAI Whisper API."""
    
    # Check if we need to extract audio from video container
    actual_filepath = filepath
    mime_type = recording.mime_type if recording else None
    
    # Detect video containers
    is_video = False
    if mime_type:
        is_video = mime_type.startswith('video/')
    else:
        # Fallback to extension-based detection
        is_video = filepath.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv', '.3gp'))
    
    if is_video:
        logger.info(f"Video container detected for Whisper transcription, extracting audio...")
        try:
            # Extract audio from video
            audio_filepath, audio_mime_type = extract_audio_from_video(filepath, 'wav')
            actual_filepath = audio_filepath
            
            # Update recording with extracted audio path and new MIME type if recording exists
            if recording:
                recording.audio_path = audio_filepath
                recording.mime_type = audio_mime_type
                db.session.commit()
            
            logger.info(f"Audio extracted successfully for Whisper: {audio_filepath}")
        except Exception as e:
            logger.error(f"Failed to extract audio from video for Whisper: {str(e)}")
            if recording:
                recording.status = 'FAILED'
                recording.error_msg = f"Audio extraction failed: {str(e)}"
                db.session.commit()
            raise Exception(f"Audio extraction failed: {str(e)}")
    
    # List of formats supported by Whisper API
    WHISPER_SUPPORTED_FORMATS = ['flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm']
    
    # Check if the file format needs conversion
    file_ext = os.path.splitext(actual_filepath)[1].lower().lstrip('.')
    converted_filepath = None
    
    try:
        with open(actual_filepath, 'rb') as audio_file:
            transcription_client = OpenAI(
                api_key=transcription_api_key,
                base_url=transcription_base_url,
                http_client=http_client_no_proxy
            )
            whisper_model = os.environ.get("WHISPER_MODEL", "Systran/faster-distil-whisper-large-v3")
            
            user_transcription_language = None
            if recording and recording.owner:
                user_transcription_language = recording.owner.transcription_language
            
            transcription_language = user_transcription_language

            transcription_params = {
                "model": whisper_model,
                "file": audio_file
            }

            if transcription_language:
                transcription_params["language"] = transcription_language
                logger.info(f"Using transcription language: {transcription_language}")
            else:
                logger.info("Transcription language not set, using auto-detection or service default.")

            transcript = transcription_client.audio.transcriptions.create(**transcription_params)
            return transcript.text
            
    except Exception as e:
        # Check if it's a format error
        error_message = str(e)
        if "Invalid file format" in error_message or "Supported formats" in error_message:
            logger.warning(f"Unsupported audio format '{file_ext}' detected, converting to MP3...")
            
            # Convert to MP3
            import tempfile
            temp_mp3_filepath = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_mp3:
                    temp_mp3_filepath = temp_mp3.name
                
                # Use ffmpeg to convert to MP3 with consistent settings
                subprocess.run(
                    ['ffmpeg', '-i', actual_filepath, '-y', '-acodec', 'libmp3lame', '-b:a', '128k', '-ar', '44100', temp_mp3_filepath],
                    check=True,
                    capture_output=True
                )
                logger.info(f"Successfully converted {actual_filepath} to MP3 format")
                converted_filepath = temp_mp3_filepath
                
                # Retry transcription with converted file
                with open(converted_filepath, 'rb') as audio_file:
                    transcription_client = OpenAI(
                        api_key=transcription_api_key,
                        base_url=transcription_base_url,
                        http_client=http_client_no_proxy
                    )
                    
                    transcription_params = {
                        "model": whisper_model,
                        "file": audio_file
                    }

                    if transcription_language:
                        transcription_params["language"] = transcription_language

                    transcript = transcription_client.audio.transcriptions.create(**transcription_params)
                    return transcript.text
                    
            finally:
                # Clean up temporary converted file
                if converted_filepath and os.path.exists(converted_filepath):
                    try:
                        os.unlink(converted_filepath)
                        logger.info(f"Cleaned up temporary converted file: {converted_filepath}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up temporary file {converted_filepath}: {cleanup_error}")
        else:
            # Re-raise if it's not a format error
            raise


def transcribe_with_chunking(app_context, recording_id, filepath, filename_for_asr):
    """Transcribe a large audio file using chunking."""
    import tempfile
    
    with app_context:
        recording = db.session.get(Recording, recording_id)
        if not recording:
            raise ValueError(f"Recording {recording_id} not found")
    
    # Create temporary directory for chunks
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create chunks
            logger.info(f"Creating chunks for large file: {filepath}")
            chunks = chunking_service.create_chunks(filepath, temp_dir)
            
            if not chunks:
                raise ChunkProcessingError("No chunks were created from the audio file")
            
            logger.info(f"Created {len(chunks)} chunks, processing each with Whisper API...")
            
            # Process each chunk with proper timeout and retry handling
            chunk_results = []
            
            # Create HTTP client with proper timeouts
            timeout_config = httpx.Timeout(
                connect=30.0,    # 30 seconds to establish connection
                read=300.0,      # 5 minutes to read response (for large audio files)
                write=60.0,      # 1 minute to write request
                pool=10.0        # 10 seconds to get connection from pool
            )
            
            http_client_with_timeout = httpx.Client(
                verify=True,
                timeout=timeout_config,
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2)
            )
            
            transcription_client = OpenAI(
                api_key=transcription_api_key,
                base_url=transcription_base_url,
                http_client=http_client_with_timeout,
                max_retries=3,  # Increased retries for better reliability
                timeout=300.0   # 5 minute timeout for API calls
            )
            whisper_model = os.environ.get("WHISPER_MODEL", "Systran/faster-distil-whisper-large-v3")
            
            # Get user language preference
            user_transcription_language = None
            with app_context:
                recording = db.session.get(Recording, recording_id)
                if recording and recording.owner:
                    user_transcription_language = recording.owner.transcription_language
            
            for i, chunk in enumerate(chunks):
                max_chunk_retries = 3
                chunk_retry_count = 0
                chunk_success = False
                
                while chunk_retry_count < max_chunk_retries and not chunk_success:
                    try:
                        retry_suffix = f" (retry {chunk_retry_count + 1}/{max_chunk_retries})" if chunk_retry_count > 0 else ""
                        logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk['filename']} ({chunk['size_mb']:.1f}MB){retry_suffix}")
                        
                        # Log detailed timing for each step
                        step_start_time = time.time()
                        
                        # Step 1: File opening
                        file_open_start = time.time()
                        with open(chunk['path'], 'rb') as chunk_file:
                            file_open_time = time.time() - file_open_start
                            logger.info(f"Chunk {i+1}: File opened in {file_open_time:.2f}s")
                            
                            # Step 2: Prepare transcription parameters
                            param_start = time.time()
                            transcription_params = {
                                "model": whisper_model,
                                "file": chunk_file
                            }
                            
                            if user_transcription_language:
                                transcription_params["language"] = user_transcription_language
                            
                            param_time = time.time() - param_start
                            logger.info(f"Chunk {i+1}: Parameters prepared in {param_time:.2f}s")
                            
                            # Step 3: API call with detailed timing
                            api_start = time.time()
                            logger.info(f"Chunk {i+1}: Starting API call to {transcription_base_url}")
                            
                            # Log connection details
                            logger.info(f"Chunk {i+1}: Using timeout config - connect: 30s, read: 300s, write: 60s")
                            logger.info(f"Chunk {i+1}: Max retries: 2, API timeout: 300s")
                            
                            try:
                                transcript = transcription_client.audio.transcriptions.create(**transcription_params)
                            except Exception as chunk_error:
                                # Check if it's a format error (unlikely for chunks since they're MP3, but handle it)
                                error_msg = str(chunk_error)
                                if "Invalid file format" in error_msg or "Supported formats" in error_msg:
                                    logger.warning(f"Chunk {i+1} format issue, attempting conversion...")
                                    # Convert chunk to MP3 if needed
                                    import tempfile
                                    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_mp3:
                                        temp_mp3_path = temp_mp3.name
                                    try:
                                        subprocess.run(
                                            ['ffmpeg', '-i', chunk['path'], '-y', '-acodec', 'libmp3lame', '-b:a', '128k', '-ar', '44100', temp_mp3_path],
                                            check=True,
                                            capture_output=True
                                        )
                                        with open(temp_mp3_path, 'rb') as converted_chunk:
                                            transcription_params['file'] = converted_chunk
                                            transcript = transcription_client.audio.transcriptions.create(**transcription_params)
                                    finally:
                                        if os.path.exists(temp_mp3_path):
                                            os.unlink(temp_mp3_path)
                                else:
                                    raise
                            
                            api_time = time.time() - api_start
                            logger.info(f"Chunk {i+1}: API call completed in {api_time:.2f}s")
                            
                            # Step 4: Process response
                            response_start = time.time()
                            chunk_result = {
                                'index': chunk['index'],
                                'start_time': chunk['start_time'],
                                'end_time': chunk['end_time'],
                                'duration': chunk['duration'],
                                'size_mb': chunk['size_mb'],
                                'transcription': transcript.text,
                                'filename': chunk['filename'],
                                'processing_time': api_time  # Store the actual API processing time
                            }
                            chunk_results.append(chunk_result)
                            response_time = time.time() - response_start
                            
                            total_time = time.time() - step_start_time
                            logger.info(f"Chunk {i+1}: Response processed in {response_time:.2f}s")
                            logger.info(f"Chunk {i+1}: Total processing time: {total_time:.2f}s")
                            logger.info(f"Chunk {i+1} transcribed successfully: {len(transcript.text)} characters")
                            chunk_success = True
                            
                    except Exception as chunk_error:
                        chunk_retry_count += 1
                        error_msg = str(chunk_error)
                        
                        if chunk_retry_count < max_chunk_retries:
                            # Determine wait time based on error type
                            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                                wait_time = 30  # 30 seconds for timeout errors
                            elif "rate limit" in error_msg.lower():
                                wait_time = 60  # 1 minute for rate limit errors
                            else:
                                wait_time = 15  # 15 seconds for other errors
                            
                            logger.warning(f"Chunk {i+1} failed (attempt {chunk_retry_count}/{max_chunk_retries}): {chunk_error}. Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Chunk {i+1} failed after {max_chunk_retries} attempts: {chunk_error}")
                            # Add failed chunk to results
                            chunk_result = {
                                'index': chunk['index'],
                                'start_time': chunk['start_time'],
                                'end_time': chunk['end_time'],
                                'transcription': f"[Chunk {i+1} transcription failed after {max_chunk_retries} attempts: {str(chunk_error)}]",
                                'filename': chunk['filename']
                            }
                            chunk_results.append(chunk_result)
                
                # Add small delay between chunks to avoid overwhelming the API
                if i < len(chunks) - 1:  # Don't delay after the last chunk
                    time.sleep(2)
            
            # Merge transcriptions
            logger.info(f"Merging {len(chunk_results)} chunk transcriptions...")
            merged_transcription = chunking_service.merge_transcriptions(chunk_results)
            
            if not merged_transcription.strip():
                raise ChunkProcessingError("Merged transcription is empty")
            
            # Log detailed performance statistics and analysis
            chunking_service.log_processing_statistics(chunk_results)
            
            # Get performance recommendations
            recommendations = chunking_service.get_performance_recommendations(chunk_results)
            if recommendations:
                logger.info("=== PERFORMANCE RECOMMENDATIONS ===")
                for i, rec in enumerate(recommendations, 1):
                    logger.info(f"{i}. {rec}")
                logger.info("=== END RECOMMENDATIONS ===")
            
            logger.info(f"Chunked transcription completed. Final length: {len(merged_transcription)} characters")
            return merged_transcription
            
        except Exception as e:
            logger.error(f"Chunking transcription failed for {filepath}: {e}")
            # Clean up chunks if they exist
            if 'chunks' in locals():
                chunking_service.cleanup_chunks(chunks)
            raise ChunkProcessingError(f"Chunked transcription failed: {str(e)}")
        finally:
            # Cleanup is handled by tempfile.TemporaryDirectory context manager
            pass