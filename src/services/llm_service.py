"""
LLM Service for LecApp.

This module handles all interactions with Language Learning Models including:
- API client setup and configuration
- Completion calls (streaming and non-streaming)
- Title generation
- Summary generation
- Event extraction
- Speaker identification
- Embedding generation and semantic search
"""

import os
import sys
import json
import re
import logging
from datetime import datetime
from openai import OpenAI
import httpx

from src.extensions import db
from src.models import Recording, Event, TranscriptChunk, SystemSetting
from src.utils import (
    safe_json_loads, format_transcription_for_llm,
    clean_llm_response, format_api_error_message
)

# Optional imports for embedding functionality
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    EMBEDDINGS_AVAILABLE = False
    # Create dummy classes to prevent import errors
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, *args, **kwargs):
            return []
    
    np = None
    cosine_similarity = None

# Get logger
logger = logging.getLogger(__name__)

# --- API Configuration ---
TEXT_MODEL_API_KEY = os.environ.get("TEXT_MODEL_API_KEY")
TEXT_MODEL_BASE_URL = os.environ.get("TEXT_MODEL_BASE_URL", "https://openrouter.ai/api/v1")
if TEXT_MODEL_BASE_URL:
    TEXT_MODEL_BASE_URL = TEXT_MODEL_BASE_URL.split('#')[0].strip()
TEXT_MODEL_NAME = os.environ.get("TEXT_MODEL_NAME", "openai/gpt-3.5-turbo")

# Set up HTTP client with custom headers for OpenRouter app identification
app_headers = {
    "HTTP-Referer": "https://github.com/baveshm/LecApp",
    "X-Title": "LecApp - AI Audio Transcription",
    "User-Agent": "LecApp/1.0 (https://github.com/baveshm/LecApp)"
}

http_client_no_proxy = httpx.Client(
    verify=True,
    headers=app_headers
)

try:
    # Always attempt to create client - use API key if provided, otherwise use placeholder
    api_key = TEXT_MODEL_API_KEY or "not-needed"
    client = OpenAI(
        api_key=api_key,
        base_url=TEXT_MODEL_BASE_URL,
        http_client=http_client_no_proxy
    )
    logger.info(f"LLM client initialized for endpoint: {TEXT_MODEL_BASE_URL}. Using model: {TEXT_MODEL_NAME}")
    if "openrouter" in TEXT_MODEL_BASE_URL.lower():
        logger.info("OpenRouter integration: App identification headers added for visibility in logs")
except Exception as client_init_e:
    logger.error(f"Failed to initialize LLM client: {client_init_e}", exc_info=True)
    client = None


# --- Core LLM Functions ---

def call_llm_completion(messages, temperature=0.7, response_format=None, stream=False, max_tokens=None):
    """
    Centralized function for LLM API calls with proper error handling and logging.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature (0-1)
        response_format: Optional response format dict (e.g., {"type": "json_object"})
        stream: Whether to stream the response
        max_tokens: Optional maximum tokens to generate
        
    Returns:
        OpenAI completion object or generator (if streaming)
    """
    if not client:
        raise ValueError("LLM client not initialized")
    
    if not TEXT_MODEL_API_KEY:
        raise ValueError("TEXT_MODEL_API_KEY not configured")
    
    try:
        completion_args = {
            "model": TEXT_MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        
        if response_format:
            completion_args["response_format"] = response_format
            
        if max_tokens:
            completion_args["max_tokens"] = max_tokens
            
        return client.chat.completions.create(**completion_args)
        
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        raise


def handle_openai_api_error(e, context="operation"):
    """Handle OpenAI API errors and return user-friendly messages."""
    error_str = str(e)
    return format_api_error_message(error_str)


# --- Embedding Functions ---

_embedding_model = None

def get_embedding_model():
    """Get or initialize the sentence transformer model."""
    global _embedding_model
    
    if not EMBEDDINGS_AVAILABLE:
        return None
        
    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return None
    return _embedding_model


def chunk_transcription(transcription, max_chunk_length=500, overlap=50):
    """
    Split transcription into overlapping chunks for better context retrieval.
    
    Args:
        transcription (str): The full transcription text
        max_chunk_length (int): Maximum characters per chunk
        overlap (int): Character overlap between chunks
    
    Returns:
        list: List of text chunks
    """
    if not transcription or len(transcription) <= max_chunk_length:
        return [transcription] if transcription else []
    
    chunks = []
    start = 0
    
    while start < len(transcription):
        end = start + max_chunk_length
        
        # Try to break at sentence boundaries
        if end < len(transcription):
            # Look for sentence endings within the last 100 characters
            sentence_end = -1
            for i in range(max(0, end - 100), end):
                if transcription[i] in '.!?':
                    # Check if it's not an abbreviation
                    if i + 1 < len(transcription) and transcription[i + 1].isspace():
                        sentence_end = i + 1
            
            if sentence_end > start:
                end = sentence_end
        
        chunk = transcription[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = max(start + 1, end - overlap)
        
        # Prevent infinite loop
        if start >= len(transcription):
            break
    
    return chunks


def generate_embeddings(texts):
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts (list): List of text strings
    
    Returns:
        list: List of embedding vectors as numpy arrays, or empty list if embeddings unavailable
    """
    if not EMBEDDINGS_AVAILABLE:
        logger.warning("Embeddings not available - skipping embedding generation")
        return []
        
    model = get_embedding_model()
    if not model or not texts:
        return []
    
    try:
        embeddings = model.encode(texts)
        return [embedding.astype(np.float32) for embedding in embeddings]
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return []


def serialize_embedding(embedding):
    """Convert numpy array to binary for database storage."""
    if embedding is None or not EMBEDDINGS_AVAILABLE:
        return None
    return embedding.tobytes()


def deserialize_embedding(binary_data):
    """Convert binary data back to numpy array."""
    if binary_data is None or not EMBEDDINGS_AVAILABLE:
        return None
    return np.frombuffer(binary_data, dtype=np.float32)


def process_recording_chunks(recording_id):
    """
    Process a recording by creating chunks and generating embeddings.
    This should be called after a recording is transcribed.
    """
    try:
        recording = db.session.get(Recording, recording_id)
        if not recording or not recording.transcription:
            return False
        
        # Delete existing chunks for this recording
        TranscriptChunk.query.filter_by(recording_id=recording_id).delete()
        
        # Create chunks
        chunks = chunk_transcription(recording.transcription)
        
        if not chunks:
            return True
        
        # Generate embeddings
        embeddings = generate_embeddings(chunks)
        
        # Store chunks in database
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk = TranscriptChunk(
                recording_id=recording_id,
                user_id=recording.user_id,
                chunk_index=i,
                content=chunk_text,
                embedding=serialize_embedding(embedding) if embedding is not None else None
            )
            db.session.add(chunk)
        
        db.session.commit()
        logger.info(f"Created {len(chunks)} chunks for recording {recording_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing chunks for recording {recording_id}: {e}")
        db.session.rollback()
        return False


def basic_text_search_chunks(user_id, query, filters=None, top_k=5):
    """
    Basic text search fallback when embeddings are not available.
    Uses simple text matching instead of semantic search.
    """
    try:
        # Build base query for chunks
        chunks_query = TranscriptChunk.query.filter_by(user_id=user_id)
        
        # Apply filters if provided
        if filters:
            if filters.get('tag_ids'):
                from src.models import RecordingTag
                chunks_query = chunks_query.join(Recording).join(
                    RecordingTag, Recording.id == RecordingTag.recording_id
                ).filter(RecordingTag.tag_id.in_(filters['tag_ids']))
            
            if filters.get('speaker_names'):
                # Filter by participants field in recordings instead of chunk speaker_name
                if not any(hasattr(desc, 'name') and desc.name == 'recording' for desc in chunks_query.column_descriptions):
                    chunks_query = chunks_query.join(Recording)
                
                # Build OR conditions for each speaker name in participants
                speaker_conditions = []
                for speaker_name in filters['speaker_names']:
                    speaker_conditions.append(
                        Recording.participants.ilike(f'%{speaker_name}%')
                    )
                
                chunks_query = chunks_query.filter(db.or_(*speaker_conditions))
                logger.info(f"Applied speaker filter for: {filters['speaker_names']}")
            
            if filters.get('recording_ids'):
                chunks_query = chunks_query.filter(
                    TranscriptChunk.recording_id.in_(filters['recording_ids'])
                )
            
            if filters.get('date_from') or filters.get('date_to'):
                chunks_query = chunks_query.join(Recording)
                if filters.get('date_from'):
                    chunks_query = chunks_query.filter(Recording.meeting_date >= filters['date_from'])
                if filters.get('date_to'):
                    chunks_query = chunks_query.filter(Recording.meeting_date <= filters['date_to'])
        
        # Simple text search - split query into words and search for them
        query_words = query.lower().split()
        if query_words:
            # Create a filter that matches any of the query words in the content
            text_conditions = []
            for word in query_words:
                text_conditions.append(TranscriptChunk.content.ilike(f'%{word}%'))
            
            # Combine conditions with OR
            from sqlalchemy import or_
            chunks_query = chunks_query.filter(or_(*text_conditions))
        
        # Get chunks and return with dummy similarity scores
        chunks = chunks_query.limit(top_k).all()
        
        # Return chunks with dummy similarity scores (1.0 for found chunks)
        return [(chunk, 1.0) for chunk in chunks]
        
    except Exception as e:
        logger.error(f"Error in basic text search: {e}")
        return []


def semantic_search_chunks(user_id, query, filters=None, top_k=5):
    """
    Perform semantic search on transcript chunks with filtering.
    
    Args:
        user_id (int): User ID for permission filtering
        query (str): Search query
        filters (dict): Optional filters for tags, speakers, dates, recording_ids
        top_k (int): Number of top chunks to return
    
    Returns:
        list: List of relevant chunks with similarity scores
    """
    try:
        # If embeddings are not available, fall back to basic text search
        if not EMBEDDINGS_AVAILABLE:
            logger.info("Embeddings not available - using basic text search as fallback")
            return basic_text_search_chunks(user_id, query, filters, top_k)
        
        # Generate embedding for the query
        model = get_embedding_model()
        if not model:
            return basic_text_search_chunks(user_id, query, filters, top_k)
        
        query_embedding = model.encode([query])[0]
        
        # Build base query for chunks with eager loading of recording relationship
        from sqlalchemy.orm import joinedload
        from src.models import RecordingTag
        chunks_query = TranscriptChunk.query.options(joinedload(TranscriptChunk.recording)).filter_by(user_id=user_id)
        
        # Apply filters if provided
        if filters:
            if filters.get('tag_ids'):
                # Join with recordings that have specified tags
                chunks_query = chunks_query.join(Recording).join(
                    RecordingTag, Recording.id == RecordingTag.recording_id
                ).filter(RecordingTag.tag_id.in_(filters['tag_ids']))
            
            if filters.get('speaker_names'):
                # Filter by participants field in recordings instead of chunk speaker_name
                if not any(hasattr(desc, 'name') and desc.name == 'recording' for desc in chunks_query.column_descriptions):
                    chunks_query = chunks_query.join(Recording)
                
                # Build OR conditions for each speaker name in participants
                speaker_conditions = []
                for speaker_name in filters['speaker_names']:
                    speaker_conditions.append(
                        Recording.participants.ilike(f'%{speaker_name}%')
                    )
                
                chunks_query = chunks_query.filter(db.or_(*speaker_conditions))
                logger.info(f"Applied speaker filter for: {filters['speaker_names']}")
            
            if filters.get('recording_ids'):
                chunks_query = chunks_query.filter(
                    TranscriptChunk.recording_id.in_(filters['recording_ids'])
                )
            
            if filters.get('date_from') or filters.get('date_to'):
                chunks_query = chunks_query.join(Recording)
                if filters.get('date_from'):
                    chunks_query = chunks_query.filter(Recording.meeting_date >= filters['date_from'])
                if filters.get('date_to'):
                    chunks_query = chunks_query.filter(Recording.meeting_date <= filters['date_to'])
        
        # Get chunks that have embeddings
        chunks = chunks_query.filter(TranscriptChunk.embedding.isnot(None)).all()
        
        if not chunks:
            return []
        
        # Calculate similarities
        chunk_similarities = []
        for chunk in chunks:
            try:
                chunk_embedding = deserialize_embedding(chunk.embedding)
                if chunk_embedding is not None:
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        chunk_embedding.reshape(1, -1)
                    )[0][0]
                    chunk_similarities.append((chunk, float(similarity)))
            except Exception as e:
                logger.warning(f"Error calculating similarity for chunk {chunk.id}: {e}")
                continue
        
        # Sort by similarity and return top k
        chunk_similarities.sort(key=lambda x: x[1], reverse=True)
        return chunk_similarities[:top_k]
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return []


# --- Generation Tasks ---

def generate_title_task(app_context, recording_id):
    """Generates only a title for a recording based on transcription.
    
    Args:
        app_context: Flask app context
        recording_id: ID of the recording
    """
    with app_context:
        recording = db.session.get(Recording, recording_id)
        if not recording:
            logger.error(f"Error: Recording {recording_id} not found for title generation.")
            return
            
        if client is None:
            logger.warning(f"Skipping title generation for {recording_id}: OpenRouter client not configured.")
            # Still mark as completed even if we can't generate a title
            recording.status = 'COMPLETED'
            recording.completed_at = datetime.utcnow()
            db.session.commit()
            return
            
        if not recording.transcription or len(recording.transcription.strip()) < 10:
            logger.warning(f"Transcription for recording {recording_id} is too short or empty. Skipping title generation.")
            # Still mark as completed even if we can't generate a title
            recording.status = 'COMPLETED'
            recording.completed_at = datetime.utcnow()
            db.session.commit()
            return
        
        # Get configurable transcript length limit and format transcription for LLM
        transcript_limit = SystemSetting.get_setting('transcript_length_limit', 30000)
        if transcript_limit == -1:
            raw_transcription = recording.transcription
        else:
            raw_transcription = recording.transcription[:transcript_limit]
            
        # Convert ASR JSON to clean text format
        transcript_text = format_transcription_for_llm(raw_transcription)
        
        # Get user language preference
        user_output_language = None
        if recording.owner:
            user_output_language = recording.owner.output_language
            
        language_directive = f"Please provide the title in {user_output_language}." if user_output_language else ""
        
        prompt_text = f"""Create a short title for this conversation:

{transcript_text}

Requirements:
- Maximum 8 words
- No phrases like "Discussion about" or "Meeting on"  
- Just the main topic

{language_directive}

Title:"""

        system_message_content = "You are an AI assistant that generates concise titles for audio transcriptions. Respond only with the title."
        if user_output_language:
            system_message_content += f" Ensure your response is in {user_output_language}."
        
            
        try:
            completion = call_llm_completion(
                messages=[
                    {"role": "system", "content": system_message_content},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.7,
                max_tokens=5000
            )
            
            raw_response = completion.choices[0].message.content
            reasoning = getattr(completion.choices[0].message, 'reasoning', None)
            
            # Use reasoning content if main content is empty (fallback for reasoning models)
            if not raw_response and reasoning:
                logger.info(f"Title generation for recording {recording_id}: Using reasoning field as fallback")
                # Try to extract a title from the reasoning field
                lines = reasoning.strip().split('\n')
                # Look for the last line that might be the title
                for line in reversed(lines):
                    line = line.strip()
                    if line and not line.startswith('I') and len(line.split()) <= 8:
                        raw_response = line
                        break
            
            title = clean_llm_response(raw_response) if raw_response else ""
            
            if title:
                recording.title = title
                logger.info(f"Title generated for recording {recording_id}: {title}")
            else:
                logger.warning(f"Empty title generated for recording {recording_id}")
                
        except Exception as e:
            logger.error(f"Error generating title for recording {recording_id}: {str(e)}")
            logger.error(f"Exception details:", exc_info=True)
        
        # Always set status to COMPLETED after title generation (successful or not)
        # This ensures transcription processing is marked as complete
        recording.status = 'COMPLETED'
        recording.completed_at = datetime.utcnow()
        db.session.commit()
        
        # Process chunks for semantic search after completion (if inquire mode is enabled)
        ENABLE_INQUIRE_MODE = os.environ.get('ENABLE_INQUIRE_MODE', 'false').lower() == 'true'
        if ENABLE_INQUIRE_MODE:
            try:
                process_recording_chunks(recording_id)
            except Exception as e:
                logger.error(f"Error processing chunks for completed recording {recording_id}: {e}")


def generate_summary_only_task(app_context, recording_id):
    """Generates only a summary for a recording (no title, no JSON response).
    
    Args:
        app_context: Flask app context
        recording_id: ID of the recording
    """
    with app_context:
        recording = db.session.get(Recording, recording_id)
        if not recording:
            logger.error(f"Error: Recording {recording_id} not found for summary generation.")
            return
            
        if client is None:
            logger.warning(f"Skipping summary generation for {recording_id}: OpenRouter client not configured.")
            recording.summary = "[Summary skipped: OpenRouter client not configured]"
            db.session.commit()
            return
            
        recording.status = 'SUMMARIZING'
        db.session.commit()
        
        logger.info(f"Requesting summary from OpenRouter for recording {recording_id} using model {TEXT_MODEL_NAME}...")
        
        if not recording.transcription or len(recording.transcription.strip()) < 10:
            logger.warning(f"Transcription for recording {recording_id} is too short or empty. Skipping summarization.")
            recording.summary = "[Summary skipped due to short transcription]"
            recording.status = 'COMPLETED'
            db.session.commit()
            return
        
        # Get user preferences and tag custom prompts
        user_summary_prompt = None
        user_output_language = None
        tag_custom_prompt = None
        
        # Collect all custom prompts from tags in the order they were added to this recording
        tag_custom_prompts = []
        if recording.tags:
            # Tags are now automatically ordered by the order they were added to this recording
            for tag in recording.tags:
                if tag.custom_prompt and tag.custom_prompt.strip():
                    tag_custom_prompts.append({
                        'name': tag.name,
                        'prompt': tag.custom_prompt.strip()
                    })
                    logger.info(f"Found custom prompt from tag '{tag.name}' for recording {recording_id}")
        
        # Create merged prompt if we have multiple tag prompts
        if tag_custom_prompts:
            if len(tag_custom_prompts) == 1:
                tag_custom_prompt = tag_custom_prompts[0]['prompt']
                logger.info(f"Using single custom prompt from tag '{tag_custom_prompts[0]['name']}' for recording {recording_id}")
            else:
                # Merge multiple prompts seamlessly as unified instructions
                merged_parts = []
                for tag_prompt in tag_custom_prompts:
                    merged_parts.append(tag_prompt['prompt'])
                tag_custom_prompt = "\n\n".join(merged_parts)
                tag_names = [tp['name'] for tp in tag_custom_prompts]
                logger.info(f"Combined custom prompts from {len(tag_custom_prompts)} tags in order added ({', '.join(tag_names)}) for recording {recording_id}")
        else:
            tag_custom_prompt = None
        
        if recording.owner:
            user_summary_prompt = recording.owner.summary_prompt
            user_output_language = recording.owner.output_language
        
        # Format transcription for LLM (convert JSON to clean text format like clipboard copy)
        formatted_transcription = format_transcription_for_llm(recording.transcription)
        
        # Get configurable transcript length limit
        transcript_limit = SystemSetting.get_setting('transcript_length_limit', 30000)
        if transcript_limit == -1:
            transcript_text = formatted_transcription
        else:
            transcript_text = formatted_transcription[:transcript_limit]
        
        language_directive = f"IMPORTANT: You MUST provide the summary in {user_output_language}. The entire response must be in {user_output_language}." if user_output_language else ""
        
        # Determine which summarization instructions to use
        # Priority order: tag custom prompt > user summary prompt > admin default prompt > hardcoded fallback
        summarization_instructions = ""
        if tag_custom_prompt:
            logger.info(f"Using tag custom prompt for recording {recording_id}")
            summarization_instructions = tag_custom_prompt
        elif user_summary_prompt:
            logger.info(f"Using user custom prompt for recording {recording_id}")
            summarization_instructions = user_summary_prompt
        else:
            # Get admin default prompt from system settings
            admin_default_prompt = SystemSetting.get_setting('admin_default_summary_prompt', None)
            if admin_default_prompt:
                logger.info(f"Using admin default prompt for recording {recording_id}")
                summarization_instructions = admin_default_prompt
            else:
                # Fallback to hardcoded default if admin hasn't set one
                summarization_instructions = """Generate a comprehensive summary that includes the following sections:
- **Key Issues Discussed**: A bulleted list of the main topics
- **Key Decisions Made**: A bulleted list of any decisions reached
- **Action Items**: A bulleted list of tasks assigned, including who is responsible if mentioned"""
                logger.info(f"Using hardcoded default prompt for recording {recording_id}")
        
        # Build context information
        current_date = datetime.now().strftime("%B %d, %Y")
        context_parts = []
        context_parts.append(f"Current date: {current_date}")
        
        # Add selected tags information
        if recording.tags:
            tag_names = [tag.name for tag in recording.tags]
            context_parts.append(f"Tags applied to this transcript by the user: {', '.join(tag_names)}")
        
        # Add user profile information if available
        if recording.owner:
            user_context_parts = []
            if recording.owner.name:
                user_context_parts.append(f"Name: {recording.owner.name}")
            if recording.owner.job_title:
                user_context_parts.append(f"Job title: {recording.owner.job_title}")
            if recording.owner.company:
                user_context_parts.append(f"Company: {recording.owner.company}")
            
            if user_context_parts:
                context_parts.append(f"Information about the user: {', '.join(user_context_parts)}")
        
        context_section = "Context:\n" + "\n".join(f"- {part}" for part in context_parts)
        
        # Build SYSTEM message: Initial instructions + Context + Language
        system_message_content = "You are an AI assistant that generates comprehensive summaries for meeting transcripts. Respond only with the summary in Markdown format. Do NOT use markdown code blocks (```markdown). Provide raw markdown content directly."
        system_message_content += f"\n\n{context_section}"
        if user_output_language:
            system_message_content += f"\n\nLanguage Requirement: You MUST generate the entire summary in {user_output_language}. This is mandatory."
        
        # Build USER message: Transcription + Summarization Instructions + Language Directive
        prompt_text = f"""Transcription:
\"\"\"
{transcript_text}
\"\"\"

Summarization Instructions:
{summarization_instructions}

{language_directive}"""
            
        # Debug logging: Log the complete prompt being sent to the LLM
        logger.info(f"Sending summarization prompt to LLM (length: {len(prompt_text)} chars). Set LOG_LEVEL=DEBUG to see full prompt details.")
        logger.debug(f"=== SUMMARIZATION DEBUG for recording {recording_id} ===")
        logger.debug(f"System message: {system_message_content}")
        logger.debug(f"User prompt (length: {len(prompt_text)} chars):\n{prompt_text}")
        logger.debug(f"=== END SUMMARIZATION DEBUG for recording {recording_id} ===")
            
        try:
            completion = call_llm_completion(
                messages=[
                    {"role": "system", "content": system_message_content},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.5,
                max_tokens=int(os.environ.get("SUMMARY_MAX_TOKENS", "3000"))
            )
            
            raw_response = completion.choices[0].message.content
            logger.info(f"Raw LLM response for recording {recording_id}: '{raw_response}'")
            
            summary = clean_llm_response(raw_response) if raw_response else ""
            logger.info(f"Processed summary length for recording {recording_id}: {len(summary)} characters")
            
            if summary:
                recording.summary = summary
                db.session.commit()
                logger.info(f"Summary generated successfully for recording {recording_id}")

                # Extract events if enabled for this user BEFORE marking as completed
                if recording.owner and recording.owner.extract_events:
                    extract_events_from_transcript(recording_id, formatted_transcription, summary)

                # Mark as completed AFTER event extraction
                recording.status = 'COMPLETED'
                recording.completed_at = datetime.utcnow()
                db.session.commit()
            else:
                logger.warning(f"Empty summary generated for recording {recording_id}")
                recording.summary = "[Summary not generated]"
                recording.status = 'COMPLETED'
                db.session.commit()
                
        except Exception as e:
            error_msg = handle_openai_api_error(e, "summary")
            logger.error(f"Error generating summary for recording {recording_id}: {str(e)}")
            recording.summary = error_msg
            recording.status = 'FAILED'
            db.session.commit()


def extract_events_from_transcript(recording_id, transcript_text, summary_text):
    """Extract calendar events from transcript using LLM.

    Args:
        recording_id: ID of the recording
        transcript_text: The formatted transcript text
        summary_text: The generated summary text
    """
    try:
        recording = db.session.get(Recording, recording_id)
        if not recording or not recording.owner or not recording.owner.extract_events:
            return  # Event extraction not enabled for this user

        logger.info(f"Extracting events for recording {recording_id}")

        # Build comprehensive context information
        current_date = datetime.now()
        context_parts = []

        # CRITICAL: Determine the reference date for relative date calculations
        reference_date = None
        reference_date_source = ""

        if recording.meeting_date:
            # Prefer meeting date if available
            reference_date = recording.meeting_date
            reference_date_source = "Meeting Date"
            context_parts.append(f"**MEETING DATE (use this for relative date calculations): {recording.meeting_date.strftime('%A, %B %d, %Y')}**")
        elif recording.created_at:
            # Fall back to upload date
            reference_date = recording.created_at.date()
            reference_date_source = "Upload Date (no meeting date available)"
            context_parts.append(f"**REFERENCE DATE (use this for relative date calculations): {recording.created_at.strftime('%A, %B %d, %Y')}**")

        context_parts.append(f"Today's actual date: {current_date.strftime('%A, %B %d, %Y')}")
        context_parts.append(f"Current time: {current_date.strftime('%I:%M %p')}")

        # Add additional recording context
        if recording.created_at:
            context_parts.append(f"Recording uploaded on: {recording.created_at.strftime('%B %d, %Y at %I:%M %p')}")
        if recording.meeting_date and reference_date_source == "Meeting Date":
            # Calculate days between meeting and today for context
            days_since = (current_date.date() - recording.meeting_date).days
            if days_since == 0:
                context_parts.append("This meeting happened today")
            elif days_since == 1:
                context_parts.append("This meeting happened yesterday")
            else:
                context_parts.append(f"This meeting happened {days_since} days ago")

        # Add user context for better understanding
        if recording.owner:
            user_context = []
            if recording.owner.name:
                user_context.append(f"User's name: {recording.owner.name}")
            if recording.owner.job_title:
                user_context.append(f"Job title: {recording.owner.job_title}")
            if recording.owner.company:
                user_context.append(f"Company: {recording.owner.company}")
            if user_context:
                context_parts.append("User information: " + ", ".join(user_context))

        # Add participants if available
        if recording.participants:
            context_parts.append(f"Participants in the meeting: {recording.participants}")

        context_section = "\n".join(context_parts)

        # Prepare the prompt for event extraction
        event_prompt = f"""You are analyzing a meeting transcript to extract calendar events. Use the context below to correctly interpret relative dates and times.

IMPORTANT CONTEXT:
{context_section}

INSTRUCTIONS:
1. **CRITICAL**: Use the MEETING DATE shown above as your reference point for ALL relative date calculations
2. When people say "next Wednesday" or "tomorrow" or "next week", calculate from the MEETING DATE, not today's date
3. Example: If the meeting date is September 13, 2025 and someone says "next Wednesday", that means September 17, 2025
4. If no specific time is mentioned for an event, use 09:00:00 (9 AM) as the default start time
5. Pay attention to time zones if mentioned
6. Extract ONLY events that are explicitly discussed as future appointments, meetings, or deadlines
7. Do NOT create events for past occurrences or general discussions

For each event found, extract:
- Title: A clear, concise title for the event
- Description: Brief description including context from the meeting
- Start date/time: The calculated actual date/time (in ISO format YYYY-MM-DDTHH:MM:SS, use 09:00:00 if no time specified)
- End date/time: When the event ends (if mentioned, in ISO format, default to 1 hour after start if not specified)
- Location: Where the event will take place (if mentioned)
- Attendees: List of people who should attend (if mentioned)
- Reminder minutes: How many minutes before to remind (default 15)

Transcript Summary:
{summary_text}

Transcript excerpt (for additional context):
{transcript_text[:8000]}

RESPONSE FORMAT:
Respond with a JSON object containing an "events" array. If no events are found, return a JSON object with an empty events array.

Example response:
{{
  "events": [
    {{
      "title": "Project Review Meeting",
      "description": "Quarterly review to discuss project progress and next steps as discussed in the meeting",
      "start_datetime": "2025-07-22T14:00:00",
      "end_datetime": "2025-07-22T15:30:00",
      "location": "Conference Room A",
      "attendees": ["John Smith", "Jane Doe", "Bob Johnson"],
      "reminder_minutes": 15
    }}
  ]
}}

CRITICAL RULES:
1. **BASE ALL DATE CALCULATIONS ON THE MEETING DATE PROVIDED IN THE CONTEXT ABOVE**
2. Only extract events that are FUTURE relative to the MEETING DATE (not today's date)
3. Convert all relative dates using the MEETING DATE as the reference point
4. Example: If the meeting date is September 13, 2025 (Friday) and someone says:
   - "next Wednesday" = September 17, 2025
   - "tomorrow" = September 14, 2025
   - "next week" = week of September 15-19, 2025
5. IMPORTANT: If no time is mentioned, always use 09:00:00 (9 AM) as the start time, NOT midnight
6. Include context from the discussion in the description
7. Do NOT invent or assume events not explicitly discussed
8. If unsure about a date/time, do not include that event"""

        completion = call_llm_completion(
            messages=[
                {"role": "system", "content": """You are an expert at extracting calendar events from meeting transcripts. You excel at:
1. Understanding relative date references ("next Tuesday", "tomorrow", "in two weeks") and converting them to absolute dates
2. Identifying genuine future appointments, meetings, and deadlines from conversations
3. Distinguishing between actual planned events vs. general discussions
4. Extracting participant names and meeting details accurately

You must respond with valid JSON format only."""},
                {"role": "user", "content": event_prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
            max_tokens=3000
        )

        response_content = completion.choices[0].message.content
        events_data = safe_json_loads(response_content, {}, logger)

        # Handle both {"events": [...]} and direct array format
        if isinstance(events_data, dict) and 'events' in events_data:
            events_list = events_data['events']
        elif isinstance(events_data, list):
            events_list = events_data
        else:
            events_list = []

        logger.info(f"Found {len(events_list)} events for recording {recording_id}")

        # Save events to database
        for event_data in events_list:
            try:
                # Parse dates
                start_dt = None
                end_dt = None

                if 'start_datetime' in event_data:
                    try:
                        # Try ISO format first
                        start_dt = datetime.fromisoformat(event_data['start_datetime'].replace('Z', '+00:00'))
                    except:
                        # Try other common formats
                        from dateutil import parser
                        try:
                            start_dt = parser.parse(event_data['start_datetime'])
                        except:
                            logger.warning(f"Could not parse start_datetime: {event_data['start_datetime']}")
                            continue  # Skip this event if we can't parse the date

                if 'end_datetime' in event_data and event_data['end_datetime']:
                    try:
                        end_dt = datetime.fromisoformat(event_data['end_datetime'].replace('Z', '+00:00'))
                    except:
                        from dateutil import parser
                        try:
                            end_dt = parser.parse(event_data['end_datetime'])
                        except:
                            pass  # End time is optional

                # Create event record
                event = Event(
                    recording_id=recording_id,
                    title=event_data.get('title', 'Untitled Event')[:200],
                    description=event_data.get('description', ''),
                    start_datetime=start_dt,
                    end_datetime=end_dt,
                    location=event_data.get('location', '')[:500] if event_data.get('location') else None,
                    attendees=json.dumps(event_data.get('attendees', [])) if event_data.get('attendees') else None,
                    reminder_minutes=event_data.get('reminder_minutes', 15)
                )

                db.session.add(event)
                logger.info(f"Added event '{event.title}' for recording {recording_id}")

            except Exception as e:
                logger.error(f"Error saving event for recording {recording_id}: {str(e)}")
                continue

        db.session.commit()

        # Refresh the recording to ensure events relationship is loaded
        recording = db.session.get(Recording, recording_id)
        if recording:
            db.session.refresh(recording)

    except Exception as e:
        logger.error(f"Error extracting events for recording {recording_id}: {str(e)}")
        db.session.rollback()


# --- Speaker Identification ---

def identify_speakers_from_text(transcription):
    """
    Uses an LLM to identify speakers from a transcription.
    """
    if not TEXT_MODEL_API_KEY:
        raise ValueError("TEXT_MODEL_API_KEY not configured.")

    # The transcription passed here could be JSON, so we format it.
    formatted_transcription = format_transcription_for_llm(transcription)

    # Extract existing speaker labels (e.g., SPEAKER_00, SPEAKER_01) in order of appearance
    all_labels = re.findall(r'\[(SPEAKER_\d+)\]', formatted_transcription)
    seen = set()
    speaker_labels = [x for x in all_labels if not (x in seen or seen.add(x))]
    
    if not speaker_labels:
        return {}

    # Get configurable transcript length limit
    transcript_limit = SystemSetting.get_setting('transcript_length_limit', 30000)
    if transcript_limit == -1:
        # No limit
        transcript_text = formatted_transcription
    else:
        transcript_text = formatted_transcription[:transcript_limit]

    prompt = f"""Analyze the following transcription and identify the names of the speakers. The speakers are labeled as {', '.join(speaker_labels)}. Based on the context of the conversation, determine the most likely name for each speaker label.

Transcription:
---
{transcript_text}
---

Respond with a single JSON object where keys are the speaker labels (e.g., "SPEAKER_00") and values are the identified full names. If a name cannot be determined, use the value "Unknown".

Example:
{{
  "SPEAKER_00": "John Doe",
  "SPEAKER_01": "Jane Smith",
  "SPEAKER_02": "Unknown"
}}

JSON Response:
"""

    try:
        completion = call_llm_completion(
            messages=[
                {"role": "system", "content": "You are an expert in analyzing conversation transcripts to identify speakers. Your response must be a single, valid JSON object."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        response_content = completion.choices[0].message.content
        speaker_map = safe_json_loads(response_content, {}, logger)

        # Post-process the map to replace "Unknown" with an empty string
        for speaker_label, identified_name in speaker_map.items():
            if identified_name.strip().lower() == "unknown":
                speaker_map[speaker_label] = ""
                
        return speaker_map
    except Exception as e:
        logger.error(f"Error calling LLM for speaker identification: {e}")
        raise


def identify_unidentified_speakers_from_text(transcription, unidentified_speakers):
    """
    Uses an LLM to identify only the unidentified speakers from a transcription.
    """
    if not TEXT_MODEL_API_KEY:
        raise ValueError("TEXT_MODEL_API_KEY not configured.")

    # The transcription passed here could be JSON, so we format it.
    formatted_transcription = format_transcription_for_llm(transcription)

    if not unidentified_speakers:
        return {}

    # Get configurable transcript length limit
    transcript_limit = SystemSetting.get_setting('transcript_length_limit', 30000)
    if transcript_limit == -1:
        # No limit
        transcript_text = formatted_transcription
    else:
        transcript_text = formatted_transcription[:transcript_limit]

    prompt = f"""Analyze the following conversation transcript and identify the names of the UNIDENTIFIED speakers based on the context and content of their dialogue. 

The speakers that need to be identified are: {', '.join(unidentified_speakers)}

Look for clues in the conversation such as:
- Names mentioned by other speakers when addressing someone
- Self-introductions or references to their own name
- Context clues about roles, relationships, or positions
- Any direct mentions of names in the dialogue

Here is the complete conversation transcript:

{transcript_text}

Based on the conversation above, identify the most likely real names for the unidentified speakers. Pay close attention to how speakers address each other and any names that are mentioned in the dialogue.

Respond with a single JSON object where keys are the speaker labels (e.g., "SPEAKER_01") and values are the identified full names. If a name cannot be determined from the conversation context, use an empty string "".

Example format:
{{
  "SPEAKER_01": "Jane Smith",
  "SPEAKER_03": "Bob Johnson",
  "SPEAKER_05": ""
}}

JSON Response:
"""

    try:
        completion = call_llm_completion(
            messages=[
                {"role": "system", "content": "You are an expert in analyzing conversation transcripts to identify speakers based on contextual clues in the dialogue. Analyze the conversation carefully to find names mentioned when speakers address each other or introduce themselves. Your response must be a single, valid JSON object containing only the requested speaker identifications."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        response_content = completion.choices[0].message.content
        speaker_map = safe_json_loads(response_content, {}, logger)

        # Post-process the map to replace "Unknown" with an empty string
        for speaker_label, identified_name in speaker_map.items():
            if identified_name and identified_name.strip().lower() in ["unknown", "n/a", "not available", "unclear"]:
                speaker_map[speaker_label] = ""
                
        return speaker_map
    except Exception as e:
        logger.error(f"Error calling LLM for speaker identification: {e}")
        raise
