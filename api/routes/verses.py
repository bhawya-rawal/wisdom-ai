import json
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from database.connection import get_session
from models.db_models import User, Verse, UsageLog
from schemas.api_schemas import DailyVerseResponse, SaveVerseRequest
from api.dependencies.dependencies import get_current_user
from retrieval.embeddings import embedding_service
from retrieval.faiss_engine import faiss_engine
from utils.media_helpers import generate_tts, generate_verse_image

router = APIRouter(tags=["Scriptural Verses"])

def _get_daily_verse_for_user(user: User, session: Session) -> Verse:
    """Helper to query the vector engine for a relevant daily verse based on the user's last mood"""
    mood = user.last_mood or "neutral"
    
    # Generate query embedding
    q_emb = embedding_service.get_embedding("daily verse guidance")
    
    # Avoid recently shown verses
    recent_verses = set()
    if user.recent_verses:
        try:
            recent_data = json.loads(user.recent_verses)
            for m_key, v_list in recent_data.items():
                recent_verses.update(v_list)
        except Exception:
            pass
            
    # Search FAISS index
    candidates = faiss_engine.search(q_emb, top_k=10, mood=mood)
    
    # Filter out recent verses
    selection = None
    for cand in candidates:
        if cand["verse_id"] not in recent_verses:
            selection = cand
            break
            
    # Fallback to top candidate if all are recently seen, or fallback to database query
    if not selection:
        if candidates:
            selection = candidates[0]
        else:
            # Absolute fallback from DB
            db_verse = session.exec(select(Verse)).first()
            if not db_verse:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database has no scriptures seeded."
                )
            return db_verse
            
    # Find full Verse object in DB
    verse_obj = session.exec(select(Verse).where(Verse.verse_id == selection["verse_id"])).one_or_none()
    return verse_obj if verse_obj else session.exec(select(Verse)).first()

@router.get("/daily-verse", response_model=DailyVerseResponse)
def daily_verse(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Retrieve today's personalized daily verse based on last recorded mood, creating TTS audio and image cards"""
    verse = _get_daily_verse_for_user(user, session)
    
    # Generate TTS and Image url assets
    audio_url = generate_tts(verse.text, verse.verse_id)
    image_url = generate_verse_image(verse.text, verse.verse_id)
    
    # Track usage log
    session.add(UsageLog(user_id=user.id, mood=user.last_mood or "neutral", endpoint="daily-verse"))
    session.commit()
    
    return DailyVerseResponse(
        verse_id=verse.verse_id,
        text=verse.text,
        source=verse.source,
        audio_url=audio_url,
        image_url=image_url
    )

@router.post("/save-verse")
def save_verse(request: SaveVerseRequest, user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Save a verse to the user's bookmarks list"""
    db_user = session.get(User, user.id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
        
    saved_verses = json.loads(db_user.saved_verses or "[]")
    
    if request.verse_id not in saved_verses:
        saved_verses.append(request.verse_id)
        db_user.saved_verses = json.dumps(saved_verses)
        session.add(db_user)
        session.commit()
        return {"success": True, "message": "Verse saved successfully"}
        
    return {"success": True, "message": "Verse already saved"}

@router.post("/save-verse-from-daily")
def save_verse_from_daily(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Bookmark today's generated daily verse"""
    db_user = session.get(User, user.id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
        
    verse = _get_daily_verse_for_user(db_user, session)
    saved_verses = json.loads(db_user.saved_verses or "[]")
    
    if verse.verse_id not in saved_verses:
        saved_verses.append(verse.verse_id)
        db_user.saved_verses = json.dumps(saved_verses)
        session.add(db_user)
        session.commit()
        return {"success": True, "message": f"Daily verse '{verse.verse_id}' saved successfully"}
        
    return {"success": True, "message": "Daily verse already saved"}

@router.get("/daily-verse-with-save")
def daily_verse_with_save(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Retrieve today's daily verse along with a boolean indicating if it is already bookmarked"""
    verse = _get_daily_verse_for_user(user, session)
    saved_verses = json.loads(user.saved_verses or "[]")
    is_saved = verse.verse_id in saved_verses
    
    audio_url = generate_tts(verse.text, verse.verse_id)
    image_url = generate_verse_image(verse.text, verse.verse_id)
    
    # Track usage log
    session.add(UsageLog(user_id=user.id, mood=user.last_mood or "neutral", endpoint="daily-verse"))
    session.commit()
    
    return {
        "verse_id": verse.verse_id,
        "text": verse.text,
        "source": verse.source,
        "audio_url": audio_url,
        "image_url": image_url,
        "is_saved": is_saved
    }

@router.get("/my-saved-verses")
def my_saved_verses(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Retrieve details of all bookmarked verses for the authenticated user"""
    saved_ids = json.loads(user.saved_verses or "[]")
    if not saved_ids:
        return []
        
    # Fetch details for each saved verse
    stmt = select(Verse).where(Verse.verse_id.in_(saved_ids))
    results = session.exec(stmt).all()
    
    return [
        {
            "verse_id": v.verse_id,
            "text": v.text,
            "source": v.source,
            "mood_tags": json.loads(v.mood_tags or "[]")
        }
        for v in results
    ]
