import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlmodel import Session, select, func

from database.connection import get_session
from models.db_models import User, ChatSummary, UsageLog, FlaggedComment, Verse
from api.dependencies.dependencies import get_current_user, get_current_admin
from retrieval.faiss_engine import faiss_engine
from services.llm.llm_service import llm_service

router = APIRouter(tags=["Admin Panel"])

class UserUpdateRequest(BaseModel):
    is_admin: Optional[bool] = None
    name: Optional[str] = None

def serialize_flagged_comment(comment: FlaggedComment) -> Dict[str, Any]:
    return {
        "id": comment.id,
        "verse_id": comment.verse_id,
        "comment": comment.comment,
        "user_name": comment.user_name,
        "user_email": comment.user_email,
        "created_at": comment.created_at.isoformat(),
        "status": comment.status,
        "reason": comment.reason
    }

def ensure_sample_flagged_comments(session: Session):
    existing = session.exec(select(FlaggedComment).limit(1)).first()
    if existing:
        return
    samples = [
        FlaggedComment(
            user_name="Demo User",
            user_email="demo@example.com",
            verse_id="Gita_2.47",
            comment="This verse seems confusing to me, can someone clarify?",
            reason="review_requested"
        ),
        FlaggedComment(
            user_name="Test Moderator",
            user_email="moderator@example.com",
            verse_id="Psalm_23.1",
            comment="I think this content might be duplicated elsewhere.",
            reason="possible_duplicate"
        )
    ]
    session.add_all(samples)
    session.commit()

@router.get("/admin/check")
def admin_check(user: User = Depends(get_current_admin)):
    """Simple verification route to check if the current user is an admin"""
    return {"admin": True, "user_id": user.uuid, "name": user.name}

@router.get("/admin/analytics")
def admin_analytics(admin: User = Depends(get_current_admin), session: Session = Depends(get_session)):
    """Fetch user engagement numbers, peak hours, mood distribution, and popular verse stats"""
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    active_users = session.exec(
        select(UsageLog.user_id)
        .where(UsageLog.timestamp >= thirty_days_ago)
    ).all()
    active_user_count = len(set(active_users))
    
    # Mood logs last 7 days
    week_ago = datetime.utcnow() - timedelta(days=7)
    mood_logs = session.exec(
        select(UsageLog)
        .where(UsageLog.timestamp >= week_ago)
    ).all()
    
    mood_counts = {}
    for log in mood_logs:
        if log.mood:
            mood_counts[log.mood] = mood_counts.get(log.mood, 0) + 1
            
    # Peak usage hours
    hour_counts = {}
    for log in mood_logs:
        hour = log.timestamp.hour
        hour_counts[hour] = hour_counts.get(hour, 0) + 1
    peak_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Verse popular views
    verse_counts = {}
    verse_logs = session.exec(
        select(ChatSummary.verse_id)
        .where(ChatSummary.date >= week_ago)
    ).all()
    for verse_id in verse_logs:
        if verse_id:
            verse_counts[verse_id] = verse_counts.get(verse_id, 0) + 1
    popular_verses = sorted(verse_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "active_users_last_30_days": active_user_count,
        "mood_distribution_last_7_days": mood_counts,
        "peak_usage_hours": peak_hours,
        "popular_verses": popular_verses,
        "total_users": len(session.exec(select(User)).all()),
        "total_verses": len(session.exec(select(Verse)).all())
    }

@router.get("/admin/system-health")
def admin_system_health(admin: User = Depends(get_current_admin), session: Session = Depends(get_session)):
    """System health check showing latency and status of databases and engines"""
    db_status = True
    db_latency_ms = None
    try:
        start = datetime.utcnow()
        session.exec(select(User).limit(1)).first()
        db_latency_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
    except Exception:
        db_status = False
        
    return {
        "application": True,
        "database": db_status,
        "api": True,
        "llm": True,
        "details": {
            "database_latency_ms": db_latency_ms,
            "rag_loaded": faiss_engine.index is not None,
            "llm_provider": "groq" if llm_service.use_groq else ("ollama" if llm_service.use_ollama else "none"),
            "timestamp": datetime.utcnow().isoformat()
        }
    }

@router.get("/admin/recent-activity")
def admin_recent_activity(admin: User = Depends(get_current_admin), session: Session = Depends(get_session)):
    """Fetch chronological feed of the 20 most recent user usage logs"""
    logs = session.exec(
        select(UsageLog).order_by(UsageLog.timestamp.desc()).limit(20)
    ).all()
    
    user_cache = {}
    activity = []
    for log in logs:
        if log.user_id not in user_cache:
            user_cache[log.user_id] = session.get(User, log.user_id)
        user_obj = user_cache.get(log.user_id)
        user_name = user_obj.name if user_obj else "Unknown user"
        
        if log.endpoint in ("chat", "daily-verse"):
            item_type = "chat"
            detail = f"{user_name} engaged with the AI ({log.endpoint})."
        elif log.endpoint in ("save-verse", "save-verse-from-daily"):
            item_type = "verse_saved"
            detail = f"{user_name} saved a verse for later."
        else:
            item_type = "user_login"
            detail = f"{user_name} opened the app."
            
        activity.append({
            "type": item_type,
            "message": detail,
            "timestamp": log.timestamp.isoformat()
        })
    return activity

@router.get("/admin/analytics/engagement")
def admin_engagement_metrics(
    days: int = Query(30, ge=1, le=90),
    admin: User = Depends(get_current_admin),
    session: Session = Depends(get_session)
):
    """Aggregate engagement analytics over specific time window"""
    since = datetime.utcnow() - timedelta(days=days)
    logs = session.exec(select(UsageLog).where(UsageLog.timestamp >= since)).all()
    
    event_counts = defaultdict(int)
    daily_active_users = defaultdict(set)
    
    for log in logs:
        event_name = (log.endpoint or "unknown").replace("/", "_")
        event_counts[event_name] += 1
        if log.user_id:
            day_key = log.timestamp.date().isoformat()
            daily_active_users[day_key].add(log.user_id)
            
    dau_counts = {day: len(users) for day, users in daily_active_users.items()}
    return {
        "event_counts": event_counts,
        "daily_active_users": dau_counts,
        "total_events": sum(event_counts.values())
    }

@router.get("/admin/analytics/verse-popularity")
def admin_verse_popularity(
    limit: int = Query(20, ge=1, le=100),
    admin: User = Depends(get_current_admin),
    session: Session = Depends(get_session)
):
    """Retrieve rank statistics of most frequently fetched/recommended scripture verses"""
    stmt = (
        select(ChatSummary.verse_id, func.count(ChatSummary.id).label("views"))
        .where(ChatSummary.verse_id.is_not(None))
        .group_by(ChatSummary.verse_id)
        .order_by(func.count(ChatSummary.id).desc())
        .limit(limit)
    )
    rows = session.exec(stmt).all()
    verse_ids = [row[0] for row in rows if row[0]]
    
    verses = {}
    if verse_ids:
        verse_records = session.exec(select(Verse).where(Verse.verse_id.in_(verse_ids))).all()
        verses = {v.verse_id: v for v in verse_records}
        
    response = []
    for verse_id, views in rows:
        if not verse_id:
            continue
        v_obj = verses.get(verse_id)
        response.append({
            "verse_id": verse_id,
            "views": views,
            "text": v_obj.text if v_obj else "",
            "source": v_obj.source if v_obj else "Unknown"
        })
    return response

@router.get("/admin/moderation/flagged")
def admin_flagged_comments(admin: User = Depends(get_current_admin), session: Session = Depends(get_session)):
    """Get pending user feedback/flagged comment records for review"""
    ensure_sample_flagged_comments(session)
    comments = session.exec(
        select(FlaggedComment)
        .where(FlaggedComment.status == "pending")
        .order_by(FlaggedComment.created_at.desc())
    ).all()
    return [serialize_flagged_comment(c) for c in comments]

@router.post("/admin/moderation/{comment_id}/approve")
def admin_approve_comment(
    comment_id: int, 
    admin: User = Depends(get_current_admin),
    session: Session = Depends(get_session)
):
    """Approve a flagged comment, changing its review status"""
    comment = session.get(FlaggedComment, comment_id)
    if not comment:
        raise HTTPException(status_code=404, detail="Flagged comment not found")
    comment.status = "approved"
    session.add(comment)
    session.commit()
    session.refresh(comment)
    return {"success": True, "message": "Comment approved", "comment": serialize_flagged_comment(comment)}

@router.delete("/admin/moderation/{comment_id}/delete")
def admin_delete_comment(
    comment_id: int, 
    admin: User = Depends(get_current_admin),
    session: Session = Depends(get_session)
):
    """Remove a flagged feedback comment permanently from moderation queue"""
    comment = session.get(FlaggedComment, comment_id)
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")
    session.delete(comment)
    session.commit()
    return {"success": True, "message": "Comment deleted"}

@router.get("/admin/users")
def admin_users(
    search: Optional[str] = Query(None),
    admin: User = Depends(get_current_admin),
    session: Session = Depends(get_session)
):
    """Query registered user accounts (admin only) with fuzzy match filters"""
    stmt = select(User)
    if search:
        term = f"%{search}%"
        # SQLite uses LIKE; Postgres ILIKE. Try Postgres ILIKE, fall back to SQLite LIKE.
        try:
            stmt = stmt.where((User.name.ilike(term)) | (User.email.ilike(term)))
        except Exception:
            stmt = stmt.where((User.name.like(term)) | (User.email.like(term)))
            
    users = session.exec(stmt.order_by(User.created_at.desc())).all()
    return [
        {
            "id": u.id,
            "uuid": u.uuid,
            "name": u.name,
            "email": u.email,
            "is_admin": u.is_admin,
            "created_at": u.created_at.isoformat(),
            "last_mood": u.last_mood
        }
        for u in users
    ]

@router.put("/admin/users/{user_id}")
def admin_update_user(
    user_id: int,
    update_data: UserUpdateRequest,
    admin: User = Depends(get_current_admin),
    session: Session = Depends(get_session)
):
    """Edit account configurations or toggle admin status of a user"""
    target = session.get(User, user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
        
    if admin.id == user_id and update_data.is_admin is False:
        raise HTTPException(status_code=400, detail="Cannot demote yourself from admin privileges")
        
    if update_data.is_admin is not None:
        target.is_admin = update_data.is_admin
    if update_data.name is not None:
        target.name = update_data.name
        
    target.updated_at = datetime.utcnow()
    session.add(target)
    session.commit()
    session.refresh(target)
    return {
        "id": target.id,
        "uuid": target.uuid,
        "name": target.name,
        "email": target.email,
        "is_admin": target.is_admin,
        "created_at": target.created_at.isoformat()
    }

@router.delete("/admin/users/{user_id}")
def admin_delete_user(
    user_id: int, 
    admin: User = Depends(get_current_admin),
    session: Session = Depends(get_session)
):
    """Delete a user account from database"""
    target = session.get(User, user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
        
    if admin.id == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own active admin account")
    if target.is_admin:
        raise HTTPException(status_code=400, detail="Cannot delete secondary admin accounts")
        
    session.delete(target)
    session.commit()
    return {"success": True, "message": "User deleted"}

@router.post("/admin/embedding-status")
def admin_embedding_status(admin: User = Depends(get_current_admin)):
    """Fetch file sizes and dimensions of local FAISS vector storage files"""
    info = faiss_engine.load_index()
    return {
        "loaded_from_disk": info,
        "current_embeddings_in_memory": len(faiss_engine.index_to_verse_id) if faiss_engine.index_to_verse_id else 0,
        "index_dimension": faiss_engine.dimension,
        "using_faiss_library": getattr(faiss_engine, "index", None) is not None
    }

@router.post("/admin/regenerate-embeddings")
def admin_regenerate_embeddings(admin: User = Depends(get_current_admin), session: Session = Depends(get_session)):
    """Force FAISS index rebuild, running embedding model over the entire verse database table"""
    total = faiss_engine.rebuild_index(session)
    return {"success": True, "message": "FAISS index regenerated successfully", "total_indexed": total}
