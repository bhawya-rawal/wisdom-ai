import json
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from database.connection import get_session
from models.db_models import User, ChatSummary
from schemas.api_schemas import SignupRequest, LoginRequest, TokenResponse, UserProfile
from services.auth.auth_service import auth_service
from api.dependencies.dependencies import get_current_user

router = APIRouter(tags=["Authentication"])

@router.post("/signup", response_model=TokenResponse)
def signup(request: SignupRequest, session: Session = Depends(get_session)):
    """User account registration"""
    # Check if user already exists
    existing = session.exec(select(User).where(User.email == request.email)).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email address already registered"
        )
        
    hashed_password = auth_service.hash_password(request.password)
    new_user = User(
        name=request.name,
        email=request.email,
        hashed_password=hashed_password,
        is_admin=False
    )
    
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    
    token = auth_service.create_access_token({"sub": new_user.uuid})
    
    return TokenResponse(
        access_token=token,
        user_id=new_user.uuid,
        name=new_user.name
    )

@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest, session: Session = Depends(get_session)):
    """Authenticate user credentials and issue JWT bearer token"""
    user = session.exec(select(User).where(User.email == request.email)).first()
    if not user or not auth_service.verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
        
    token = auth_service.create_access_token({"sub": user.uuid})
    
    return TokenResponse(
        access_token=token,
        user_id=user.uuid,
        name=user.name
    )

@router.get("/profile", response_model=UserProfile)
def get_profile(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Retrieve full authenticated user profile, saved scriptures, and chat history"""
    # Get user's recent chat summaries
    recent_summaries = session.exec(
        select(ChatSummary)
        .where(ChatSummary.user_id == user.id)
        .order_by(ChatSummary.date.desc())
        .limit(10)
    ).all()
    
    chat_history = []
    for summary in recent_summaries:
        chat_history.append({
            "id": summary.id,
            "date": summary.date.isoformat(),
            "mood": summary.mood,
            "summary": summary.summary,
            "verse_id": summary.verse_id
        })
        
    return UserProfile(
        user_id=user.uuid,
        name=user.name,
        email=user.email,
        last_mood=user.last_mood,
        recent_verses=json.loads(user.recent_verses or "{}"),
        saved_verses=json.loads(user.saved_verses or "[]"),
        chat_history=chat_history
    )
