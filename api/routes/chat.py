import json
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from database.connection import get_session
from models.db_models import User, ChatSummary, UsageLog
from schemas.api_schemas import ChatRequest, ChatResponse
from api.dependencies.dependencies import get_current_user
from services.rag.rag_service import rag_service
from services.memory.memory_service import memory_service
from utils.ml_models import ml_loader

router = APIRouter(tags=["Chatting Interface"])

@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest, 
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Primary chat endpoint running mood analysis, memory lookup, vector search, and safety-validated responses"""
    message = request.message
    
    # 1. Real-time Mood/Emotion detection
    mood = "neutral"
    mood_pipeline = ml_loader.mood_pipeline
    if mood_pipeline:
        try:
            mood_results = mood_pipeline(message)
            if isinstance(mood_results, list) and len(mood_results) > 0:
                actual_results = mood_results[0] if isinstance(mood_results[0], list) else mood_results
                if len(actual_results) > 0:
                    top_result = max(actual_results, key=lambda x: x['score'])
                    mood = top_result['label'].lower()
        except Exception as e:
            print(f"Mood detection inference failed: {e}")

    # 2. Persist Usage Log and update user's last mood
    db_user = session.get(User, user.id)
    if db_user:
        db_user.last_mood = mood
        db_user.updated_at = datetime.utcnow()
        session.add(UsageLog(user_id=user.id, mood=mood, endpoint="chat"))
        session.commit()
        session.refresh(db_user)

    # 3. Add user question to raw chat message history
    memory_service.add_message(user_id=user.id, role="user", content=message, session=session)

    # 4. Trigger rolling summarization check
    history_summary = memory_service.update_summary_if_needed(user_id=user.id, session=session)
    
    # 5. Load recent history for current session context window
    _, recent_messages = memory_service.get_conversation_context(user_id=user.id, session=session)

    # 6. Run the production-grade RAG pipeline (FAISS + Reranking + Citations + Confidence check)
    reply, verse_info, latencies = rag_service.answer_question(
        user_query=message,
        user_id=user.id,
        mood=mood,
        history_summary=history_summary,
        session=session
    )

    # 7. Add generated response to conversation log
    memory_service.add_message(user_id=user.id, role="assistant", content=reply, session=session)

    # 8. Compute rolling chat summary text for this turn (using BART if available, fallback to query snippet)
    summary_text = message[:200]
    summary_pipeline = ml_loader.summary_pipeline
    if summary_pipeline and len(message.split()) > 10:
        try:
            input_length = len(message.split())
            max_len = max(10, min(60, input_length // 2))
            min_len = max(5, min(10, max_len // 2))
            summary_result = summary_pipeline(message, max_length=max_len, min_length=min_len)
            if isinstance(summary_result, list) and len(summary_result) > 0:
                summary_text = summary_result[0]['summary_text']
        except Exception as e:
            print(f"BART text summarization failed: {e}")

    # 9. Create final ChatSummary log record
    session.add(ChatSummary(
        user_id=user.id,
        mood=mood,
        summary=summary_text,
        verse_id=verse_info["verse_id"]
    ))

    # 10. Update User's recent verses (keeping last 3 per mood to prevent repetition)
    recent_verses = json.loads(db_user.recent_verses or "{}")
    if mood not in recent_verses:
        recent_verses[mood] = []
    
    recent_verses[mood].append(verse_info["verse_id"])
    if len(recent_verses[mood]) > 3:
        recent_verses[mood] = recent_verses[mood][-3:]
        
    db_user.recent_verses = json.dumps(recent_verses)
    session.commit()

    return ChatResponse(
        reply=reply,
        detected_mood=mood,
        verse_id=verse_info["verse_id"],
        verse_text=verse_info["text"],
        verse_source=verse_info["source"],
        citations=verse_info["citations"],
        confidence=verse_info["confidence"],
        confidence_label=verse_info["confidence_label"],
        conversation_summary=history_summary,
        latency_ms=latencies
    )

@router.get("/chats/recent")
def get_recent_chats(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Fetch sidebar log details of recent user conversations"""
    recent_summaries = session.exec(
        select(ChatSummary)
        .where(ChatSummary.user_id == user.id)
        .order_by(ChatSummary.date.desc())
        .limit(20)
    ).all()
    
    return [
        {
            "id": summary.id,
            "date": summary.date.isoformat(),
            "mood": summary.mood,
            "summary": summary.summary,
            "verse_id": summary.verse_id
        }
        for summary in recent_summaries
    ]

@router.delete("/chats/{chat_id}")
def delete_chat(
    chat_id: int, 
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Delete a specific chat summary log"""
    # Refresh user
    db_user = session.get(User, user.id)
    if not db_user:
        raise HTTPException(status_code=401, detail="User not found")
        
    # Check ownership
    user_chat = session.exec(
        select(ChatSummary)
        .where(ChatSummary.id == chat_id)
        .where(ChatSummary.user_id == db_user.id)
    ).one_or_none()
    
    if not user_chat:
        raise HTTPException(
            status_code=403, 
            detail="Not authorized to delete this chat."
        )
        
    session.delete(user_chat)
    session.commit()
    return {"success": True, "message": "Chat deleted"}
