import logging
from datetime import datetime, timedelta
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select, func

from config.settings import settings
from database.connection import init_db, engine
from models.db_models import User, Verse, UsageLog, ChatSummary
from retrieval.faiss_engine import faiss_engine

# Middleware & Router imports
from api.middleware.middleware import setup_cors, track_latency_middleware
from api.routes.auth import router as auth_router
from api.routes.chat import router as chat_router
from api.routes.verses import router as verses_router
from api.routes.admin import router as admin_router
from api.routes.evaluation import router as evaluation_router
from services.monitoring.observability import metrics_tracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION
)

# 1. Setup Middlewares
setup_cors(app)
app.middleware("http")(track_latency_middleware)

# 2. Register Routers
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(verses_router)
app.include_router(admin_router)
app.include_router(evaluation_router)

# 3. Mount Static Files
app.mount("/media", StaticFiles(directory=settings.MEDIA_ROOT), name="media")

# 4. Lifespan Startup Event
@app.on_event("startup")
def startup_event():
    logger.info("Initializing WisdomAI Backend Components...")
    
    # Initialize SQL Database
    init_db()
    
    # Initialize / Load FAISS vector index
    try:
        with Session(engine) as session:
            # Check if index metadata files exist; if not, compile database entries
            if not faiss_engine.index_file.exists():
                logger.info("FAISS Index file missing. Performing initial index rebuild...")
                faiss_engine.rebuild_index(session)
            else:
                faiss_engine.load_index()
    except Exception as e:
        logger.error("Failed to compile or load FAISS search engine: %s", e)

# 5. Core Public Routes
@app.get("/health")
def health_check():
    """Verify application health status"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/stats/public")
def public_stats():
    """Get public statistics for landing page (no authentication required)"""
    with Session(engine) as session:
        # Total registered users
        total_users = session.exec(select(func.count(User.id))).one() or 0
        
        # Active users (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        active_user_ids = session.exec(
            select(UsageLog.user_id)
            .where(UsageLog.timestamp >= thirty_days_ago)
        ).all()
        active_users = len(set(active_user_ids)) if active_user_ids else 0
        
        # Total conversations (chat summaries)
        total_conversations = session.exec(select(func.count(ChatSummary.id))).one() or 0
        
        # Verses shared (unique verses in chat summaries)
        verses_shared = session.exec(
            select(func.count(func.distinct(ChatSummary.verse_id)))
            .where(ChatSummary.verse_id.is_not(None))
        ).one() or 0
        
        return {
            "active_users": active_users if active_users > 0 else total_users,
            "verses_shared": verses_shared,
            "conversations": total_conversations
        }

@app.get("/metrics")
def get_metrics():
    """Endpoint exposed for Prometheus metrics scraping"""
    payload, content_type = metrics_tracker.get_metrics_payload()
    return Response(content=payload, media_type=content_type)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
