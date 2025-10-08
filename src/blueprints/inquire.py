"""Inquire blueprint (Task 11)

Provides semantic search / filtering session management when ENABLE_INQUIRE_MODE is true.
Extracted endpoints:
 - GET /api/inquire/sessions
 - POST /api/inquire/sessions
 - POST /api/inquire/search
 - POST /api/inquire/chat (streaming)
 - GET /api/inquire/available_filters
"""
from __future__ import annotations

import os
import json
from datetime import datetime
from flask import Blueprint, jsonify, request, Response, current_app, render_template, flash, redirect, url_for
from flask_login import login_required, current_user

from src.extensions import db, limiter
from src.models import InquireSession, Recording, TranscriptChunk, SystemSetting
from src.utils import process_streaming_with_thinking, format_transcription_for_llm
from src.services.llm_service import call_llm_completion

from sqlalchemy import select

ENABLE_INQUIRE_MODE = os.environ.get('ENABLE_INQUIRE_MODE', 'false').lower() == 'true'

inquire_bp = Blueprint('inquire', __name__)


def _require_enabled():
    if not ENABLE_INQUIRE_MODE:
        return jsonify({'error': 'Inquire mode is not enabled'}), 403
    return None


@inquire_bp.route('/inquire', methods=['GET'])
@login_required
def inquire_page():
    """Render the inquire mode page"""
    if not ENABLE_INQUIRE_MODE:
        flash('Inquire mode is not enabled on this server.', 'warning')
        return redirect(url_for('index'))
    return render_template('inquire.html')


@inquire_bp.route('/api/inquire/sessions', methods=['GET'])
@login_required
def get_inquire_sessions():
    disabled = _require_enabled()
    if disabled:
        return disabled
    try:
        sessions = (InquireSession.query
                    .filter_by(user_id=current_user.id)
                    .order_by(InquireSession.last_used.desc())
                    .all())
        return jsonify([s.to_dict() for s in sessions])
    except Exception as e:
        current_app.logger.error(f"Error getting inquire sessions: {e}")
        return jsonify({'error': str(e)}), 500


@inquire_bp.route('/api/inquire/sessions', methods=['POST'])
@login_required
def create_inquire_session():
    disabled = _require_enabled()
    if disabled:
        return disabled
    try:
        data = request.json or {}
        filters = data.get('filters', {})
        name = data.get('name') or f"Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
        session = InquireSession(
            user_id=current_user.id,
            name=name,
            filters=json.dumps(filters),
            created_at=datetime.utcnow(),
            last_used=datetime.utcnow()
        )
        db.session.add(session)
        db.session.commit()
        return jsonify(session.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error creating inquire session: {e}")
        return jsonify({'error': str(e)}), 500


@inquire_bp.route('/api/inquire/search', methods=['POST'])
@login_required
@limiter.limit('60 per minute')
def inquire_search():
    disabled = _require_enabled()
    if disabled:
        return disabled
    try:
        from src.services.llm_service import semantic_search_chunks
        
        data = request.json or {}
        query = data.get('query', '').strip()
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Build filters from request
        filters = {}
        if data.get('filter_tags'):
            filters['tag_ids'] = data['filter_tags']
        if data.get('filter_speakers'):
            filters['speaker_names'] = data['filter_speakers']
        if data.get('filter_recording_ids'):
            filters['recording_ids'] = data['filter_recording_ids']
        if data.get('filter_date_from'):
            filters['date_from'] = datetime.fromisoformat(data['filter_date_from']).date()
        if data.get('filter_date_to'):
            filters['date_to'] = datetime.fromisoformat(data['filter_date_to']).date()
        
        # Perform semantic search
        top_k = data.get('top_k', 5)
        chunk_results = semantic_search_chunks(current_user.id, query, filters, top_k)
        
        # Format results
        results = []
        for chunk, similarity in chunk_results:
            result = chunk.to_dict()
            result['similarity'] = similarity
            result['recording_title'] = chunk.recording.title
            result['recording_meeting_date'] = f"{chunk.recording.meeting_date.isoformat()}T00:00:00" if chunk.recording.meeting_date else None
            results.append(result)
        
        return jsonify({'results': results})
        
    except Exception as e:
        current_app.logger.error(f"Error in inquire search: {e}")
        return jsonify({'error': str(e)}), 500


@inquire_bp.route('/api/inquire/chat', methods=['POST'])
@login_required
def inquire_chat():
    disabled = _require_enabled()
    if disabled:
        return disabled
    try:
        from src.services.llm_service import semantic_search_chunks, call_llm_completion
        
        data = request.json or {}
        user_message = data.get('message') or data.get('query')
        message_history = data.get('message_history', [])
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Check if LLM client is available
        if call_llm_completion is None:
            return jsonify({'error': 'Chat service is not available (LLM client not configured)'}), 503
        
        # Build filters from request
        filters = {}
        if data.get('filter_tags'):
            filters['tag_ids'] = data['filter_tags']
        if data.get('filter_speakers'):
            filters['speaker_names'] = data['filter_speakers']
        if data.get('filter_recording_ids'):
            filters['recording_ids'] = data['filter_recording_ids']
        if data.get('filter_date_from'):
            filters['date_from'] = datetime.fromisoformat(data['filter_date_from']).date()
        if data.get('filter_date_to'):
            filters['date_to'] = datetime.fromisoformat(data['filter_date_to']).date()
        
        # Debug logging
        current_app.logger.info(f"Inquire chat - User: {current_user.username}, Query: '{user_message}', Filters: {filters}")
        
        # Capture user context before generator to avoid current_user being None
        user_id = current_user.id
        user_name = current_user.name if current_user.name else "the user"
        user_title = current_user.job_title if current_user.job_title else "professional"
        user_company = current_user.company if current_user.company else "their organization"
        user_output_language = current_user.output_language if current_user.output_language else None
        
        # Enhanced query processing with enrichment and debugging
        def create_status_response(status, message):
            """Helper to create SSE status updates"""
            return f"data: {json.dumps({'status': status, 'message': message})}\n\n"
        
        def generate_enhanced_chat():
            nonlocal user_id, user_name, user_title, user_company, user_output_language, data, filters
            
            try:
                # Send initial status
                yield create_status_response('processing', 'Analyzing your query...')
                
                # Step 1: Router - Determine if RAG lookup is needed
                router_prompt = f"""Analyze this user query to determine if it requires searching through transcription content or if it's a simple formatting/clarification request.

User query: "{user_message}"

Respond with ONLY "RAG" if the query requires searching transcriptions (asking about content, conversations, specific information from recordings).
Respond with ONLY "DIRECT" if it's a formatting request, clarification about previous responses, or doesn't require searching transcriptions.

Examples:
- "What did Beth say about the budget?" → RAG
- "Can you format this in separate headings?" → DIRECT  
- "Who mentioned the timeline?" → RAG
- "Make this more structured" → DIRECT"""

                try:
                    router_response = call_llm_completion(
                        messages=[
                            {"role": "system", "content": "You are a query router. Respond with only 'RAG' or 'DIRECT'."},
                            {"role": "user", "content": router_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=10
                    )
                    
                    route_decision = router_response.choices[0].message.content.strip().upper()
                    current_app.logger.info(f"Router decision: {route_decision}")
                    
                    if route_decision == "DIRECT":
                        # Direct response without RAG lookup
                        yield create_status_response('responding', 'Generating direct response...')
                        
                        direct_prompt = f"""You are assisting {user_name}. Respond to their request directly using proper markdown formatting.

User request: "{user_message}"

Previous conversation context (if relevant):
{json.dumps(message_history[-2:] if message_history else [])}

Use proper markdown formatting including headings (##), bold (**text**), bullet points (-), etc."""

                        stream = call_llm_completion(
                            messages=[
                                {"role": "system", "content": direct_prompt},
                                {"role": "user", "content": user_message}
                            ],
                            temperature=0.7,
                            max_tokens=int(os.environ.get("CHAT_MAX_TOKENS", "2000")),
                            stream=True
                        )
                        
                        # Use helper function to process streaming with thinking tag support
                        for response in process_streaming_with_thinking(stream):
                            yield response
                        return
                        
                except Exception as e:
                    current_app.logger.warning(f"Router failed, defaulting to RAG: {e}")
                
                # Step 2: Query enrichment - generate better search terms based on user intent
                yield create_status_response('enriching', 'Enriching search query...')
                
                enrichment_prompt = f"""You are a query enhancement assistant. Given a user's question about transcribed meetings/recordings, generate 3-5 alternative search terms or phrases that would help find relevant content in a semantic search system.

User context:
- Name: {user_name}
- Title: {user_title}  
- Company: {user_company}

User question: "{user_message}"
Available context: Transcribed meetings and recordings with speakers: {', '.join(data.get('filter_speakers', []))}.

Generate search terms that would find relevant content. Focus on:
1. Key concepts and topics using the user's actual name instead of generic terms like "me"
2. Specific terminology that might be used in their professional context
3. Alternative phrasings of the question with proper names
4. Related terms that might appear in transcripts from their meetings

Examples:
- Instead of "what Beth told me" use "what Beth told {user_name}"
- Instead of "my last conversation" use "{user_name}'s conversation"
- Use their job title and company context when relevant

Respond with only a JSON array of strings: ["term1", "term2", "term3", ...]"""
                
                try:
                    enrichment_response = call_llm_completion(
                        messages=[
                            {"role": "system", "content": "You are a query enhancement assistant. Respond only with valid JSON arrays of search terms."},
                            {"role": "user", "content": enrichment_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=200
                    )
                    
                    enriched_terms = json.loads(enrichment_response.choices[0].message.content.strip())
                    current_app.logger.info(f"Enriched search terms: {enriched_terms}")
                    
                    # Combine original query with enriched terms for search
                    search_queries = [user_message] + enriched_terms[:3]  # Use original + top 3 enriched terms
                    
                except Exception as e:
                    current_app.logger.warning(f"Query enrichment failed, using original query: {e}")
                    search_queries = [user_message]
                
                # Step 3: Semantic search with multiple queries
                yield create_status_response('searching', 'Searching transcriptions...')
                
                all_chunks = []
                seen_chunk_ids = set()
                
                for query in search_queries:
                    chunk_results = semantic_search_chunks(user_id, query, filters, 8)
                    current_app.logger.info(f"Search query '{query}' returned {len(chunk_results)} chunks")
                    
                    for chunk, similarity in chunk_results:
                        if chunk and chunk.id not in seen_chunk_ids:
                            all_chunks.append((chunk, similarity))
                            seen_chunk_ids.add(chunk.id)
                
                # Sort by similarity and take top results
                all_chunks.sort(key=lambda x: x[1], reverse=True)
                chunk_results = all_chunks[:data.get('context_chunks', 8)]
                
                current_app.logger.info(f"Final chunk results: {len(chunk_results)} chunks with similarities: {[f'{s:.3f}' for _, s in chunk_results]}")
                
                # Step 4: Build context from retrieved chunks
                yield create_status_response('contextualizing', 'Building context...')
                
                # Group chunks by recording and organize properly
                recording_chunks = {}
                recording_ids_in_context = set()
                
                for chunk, similarity in chunk_results:
                    if not chunk or not chunk.recording:
                        continue
                    recording_id = chunk.recording.id
                    recording_ids_in_context.add(recording_id)
                    
                    if recording_id not in recording_chunks:
                        recording_chunks[recording_id] = {
                            'recording': chunk.recording,
                            'chunks': []
                        }
                    
                    recording_chunks[recording_id]['chunks'].append({
                        'chunk': chunk,
                        'similarity': similarity
                    })
                
                # Build organized context pieces
                context_pieces = []
                
                for recording_id, chunk_data in recording_chunks.items():
                    recording = chunk_data['recording']
                    chunks = chunk_data['chunks']
                    
                    # Sort chunks by their index to maintain chronological order
                    chunks.sort(key=lambda x: x['chunk'].chunk_index)
                    
                    # Build recording header with complete metadata
                    header = f"=== {recording.title} [Recording ID: {recording_id}] ==="
                    if recording.meeting_date:
                        header += f" ({recording.meeting_date})"
                    
                    # Add participants information
                    if recording.participants:
                        participants_list = [p.strip() for p in recording.participants.split(',') if p.strip()]
                        header += f"\\nParticipants: {', '.join(participants_list)}"
                    
                    context_piece = header + "\\n\\n"
                    
                    # Process chunks and detect non-continuity
                    prev_chunk_index = None
                    for chunk_data in chunks:
                        chunk = chunk_data['chunk']
                        similarity = chunk_data['similarity']
                        
                        # Check for non-continuity
                        if prev_chunk_index is not None and chunk.chunk_index != prev_chunk_index + 1:
                            context_piece += "\\n[... gap in transcript - non-consecutive chunks ...]\\n\\n"
                        
                        # Add speaker information if available
                        speaker_info = ""
                        if chunk.speaker_name:
                            speaker_info = f"{chunk.speaker_name}: "
                        elif chunk.start_time is not None:
                            speaker_info = f"[{chunk.start_time:.1f}s]: "
                        
                        # Add timing info if available
                        timing_info = ""
                        if chunk.start_time is not None and chunk.end_time is not None:
                            timing_info = f" [{chunk.start_time:.1f}s-{chunk.end_time:.1f}s]"
                        
                        context_piece += f"{speaker_info}{chunk.content}{timing_info} (similarity: {similarity:.3f})\\n\\n"
                        prev_chunk_index = chunk.chunk_index
                    
                    context_pieces.append(context_piece)
                
                merged_context = "\\n\\n".join(context_pieces)
                current_app.logger.info(f"Built context from {len(chunk_results)} chunks across {len(recording_chunks)} recordings")
                
                # Step 5: Generate response
                yield create_status_response('responding', 'Generating response...')
                
                # Prepare system prompt
                language_instruction = f"Please provide all your responses in {user_output_language}." if user_output_language else ""
                
                system_prompt = f"""You are an intelligent assistant helping {user_name} explore their recorded meetings and conversations. Use only the provided context to answer questions.

Context from transcriptions:
{merged_context[:30000]}

{language_instruction}

Instructions:
- Answer based only on the provided transcript context
- If the context doesn't contain the answer, say so clearly
- Use proper markdown formatting
- Include specific quotes when relevant
- Be concise but thorough"""

                messages = [{"role": "system", "content": system_prompt}]
                if message_history:
                    messages.extend(message_history)
                messages.append({"role": "user", "content": user_message})
                
                stream = call_llm_completion(
                    messages=messages,
                    temperature=0.4,
                    max_tokens=int(os.environ.get("CHAT_MAX_TOKENS", "2000")),
                    stream=True
                )
                
                for response in process_streaming_with_thinking(stream):
                    yield response
                    
            except Exception as e:
                current_app.logger.error(f"Error in enhanced chat generation: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(generate_enhanced_chat(), mimetype='text/event-stream')
        
    except Exception as e:
        current_app.logger.error(f"Error in inquire chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@inquire_bp.route('/api/inquire/available_filters', methods=['GET'])
@login_required
def inquire_available_filters():
    disabled = _require_enabled()
    if disabled:
        return disabled
    try:
        from src.models import Tag
        
        # Get user's tags
        tags = Tag.query.filter_by(user_id=current_user.id).all()
        
        # Get unique speakers from user's recordings participants field
        recordings_with_participants = Recording.query.filter_by(user_id=current_user.id).filter(
            Recording.participants.isnot(None),
            Recording.participants != ''
        ).all()
        
        speaker_names = set()
        for recording in recordings_with_participants:
            if recording.participants:
                # Split participants by comma and clean up
                participants = [p.strip() for p in recording.participants.split(',') if p.strip()]
                speaker_names.update(participants)
        
        speaker_names = sorted(list(speaker_names))
        
        # Get user's recordings for recording-specific filtering
        recordings = Recording.query.filter_by(user_id=current_user.id).filter(
            Recording.status == 'COMPLETED'
        ).order_by(Recording.created_at.desc()).all()
        
        return jsonify({
            'tags': [tag.to_dict() for tag in tags],
            'speakers': speaker_names,
            'recordings': [{'id': r.id, 'title': r.title, 'meeting_date': f"{r.meeting_date.isoformat()}T00:00:00" if r.meeting_date else None} for r in recordings]
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting available filters: {e}")
        return jsonify({'error': str(e)}), 500
