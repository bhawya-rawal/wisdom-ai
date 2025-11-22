"""
God AI - FastAPI Backend
A comprehensive spiritual AI companion backend with mood detection, 
verse recommendations, and multimedia content generation.
"""

from fastapi import FastAPI, Depends, HTTPException, status, Request, Query
from fastapi.security import OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlmodel import Field, SQLModel, Session, create_engine, select
from sqlalchemy import func
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import uuid
import os
import hashlib
import jwt
import json
import asyncio
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()
# ML imports
ML_AVAILABLE = False
pipeline = None
torch = None
try:
    import sys
    import warnings
    warnings.filterwarnings('ignore')
    from transformers import pipeline
    import torch
    ML_AVAILABLE = True
except Exception as e:
    print(f"Warning: ML libraries not available: {e}")
    import traceback
    traceback.print_exc()
    ML_AVAILABLE = False
    pipeline = None
    torch = None

# Image generation
from PIL import Image, ImageDraw, ImageFont
import textwrap

# TTS imports
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("Warning: gTTS not available, falling back to pyttsx3")
    try:
        import pyttsx3
    except ImportError:
        pyttsx3 = None

import io
import base64

# RAG integration
from rag_integration import initialize_rag, get_rag_instance

# LLM integration
from llm_service import initialize_llm, get_llm_service

# ----------------------
# CONFIG / ENV
# ----------------------

JWT_SECRET = os.getenv("JWT_SECRET", "change_this_secret_key_please")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 30  # 30 days
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./god_ai.db")
# Render gives postgres:// but SQLAlchemy 2.x requires postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
MEDIA_ROOT = os.getenv("MEDIA_ROOT", "./media")

# Create media directories
os.makedirs(MEDIA_ROOT, exist_ok=True)
os.makedirs(f"{MEDIA_ROOT}/audio", exist_ok=True)
os.makedirs(f"{MEDIA_ROOT}/images", exist_ok=True)

from models import User, ChatSummary, UsageLog, FlaggedComment, Verse

# ----------------------
# DB INITIALIZATION
# ----------------------

engine = create_engine(DATABASE_URL, echo=False)
SQLModel.metadata.create_all(engine)

# ----------------------
# UTILITIES
# ----------------------

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

# ----------------------
# ML PIPELINES
# ----------------------

# Initialize ML pipelines
if ML_AVAILABLE:
    try:
        mood_pipeline = pipeline(
            "text-classification", 
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        summary_pipeline = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn"
        )
    except Exception as e:
        print(f"Warning: Could not load ML models: {e}")
        mood_pipeline = None
        summary_pipeline = None
else:
    mood_pipeline = None
    summary_pipeline = None

# ----------------------
# TTS SERVICE
# ----------------------

def generate_tts(verse_text: str, verse_id: str) -> str:
    """Generate TTS audio file for the verse text with Hindi spiritual voice"""
    try:
        audio_filename = f"{verse_id.replace(' ', '_').replace('.', '_')}.mp3"
        audio_path = os.path.join(MEDIA_ROOT, "audio", audio_filename)
        
        # Use gTTS for better quality Hindi spiritual voice
        if GTTS_AVAILABLE:
            try:
                # Detect if text contains Hindi/Devanagari characters
                has_hindi = any('\u0900' <= char <= '\u097F' for char in verse_text)
                
                if has_hindi:
                    # Use Hindi language for Hindi text
                    tts = gTTS(text=verse_text, lang='hi', slow=True)
                else:
                    # Use English with Indian accent (co.in domain) for English text
                    # slow=True creates a more contemplative, spiritual tone
                    tts = gTTS(text=verse_text, lang='en', slow=True, tld='co.in')
                
                tts.save(audio_path)
                print(f"Generated TTS audio with gTTS: {audio_path}")
                return f"/media/audio/{audio_filename}"
            except Exception as e:
                print(f"gTTS generation failed, falling back to pyttsx3: {e}")
                import traceback
                traceback.print_exc()
                # Fall through to pyttsx3
        
        # Fallback to pyttsx3 if gTTS is not available
        if pyttsx3:
            engine = pyttsx3.init()
            
            # Configure voice properties for a deeper, more spiritual tone
            voices = engine.getProperty('voices')
            if voices:
                # Try to find a male voice (usually deeper)
                for voice in voices:
                    # Look for male voices (Windows) or deeper voices
                    if 'male' in voice.name.lower() or 'david' in voice.name.lower() or 'zira' not in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
                else:
                    # Use first available voice if no male voice found
                    engine.setProperty('voice', voices[0].id)
            
            # Slower rate for more contemplative, spiritual feel
            engine.setProperty('rate', 120)  # Slower speed (default is usually 200)
            engine.setProperty('volume', 0.95)  # Slightly higher volume
            
            # Try to set pitch lower for a deeper, more god-like voice
            try:
                engine.setProperty('pitch', 0.7)  # Lower pitch (0.5 to 2.0, default is 1.0)
            except:
                pass  # Some engines don't support pitch
            
            engine.save_to_file(verse_text, audio_path)
            engine.runAndWait()
            
            return f"/media/audio/{audio_filename}"
        else:
            print("No TTS engine available")
            return None
            
    except Exception as e:
        print(f"TTS generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# ----------------------
# IMAGE GENERATION
# ----------------------

def generate_verse_image(verse_text: str, verse_id: str) -> str:
    """Create an attractive image with the verse text"""
    try:
        # Image dimensions
        width, height = 1200, 630
        
        # Create gradient background
        img = Image.new("RGB", (width, height), color=(250, 245, 240))
        draw = ImageDraw.Draw(img)
        
        # Create gradient effect
        for y in range(height):
            color_value = int(250 - (y / height) * 30)
            draw.line([(0, y), (width, y)], fill=(color_value, color_value-5, color_value-10))
        
        # Try to load a nice font
        try:
            # Try system fonts
            font_large = ImageFont.truetype("arial.ttf", 32)
            font_medium = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except:
            try:
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
                font_small = ImageFont.load_default()
            except:
                font_large = font_medium = font_small = None
        
        # Wrap text to fit within margins
        margin = 80
        max_width = width - (margin * 2)
        
        # Split verse into lines
        words = verse_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if font_large:
                bbox = draw.textbbox((0, 0), test_line, font=font_large)
                text_width = bbox[2] - bbox[0]
            else:
                text_width = len(test_line) * 10  # Rough estimate
            
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    lines.append(word)
        
        if current_line:
            lines.append(current_line)
        
        # Draw verse text
        line_height = 45
        start_y = height // 2 - (len(lines) * line_height) // 2
        
        for i, line in enumerate(lines):
            y_pos = start_y + (i * line_height)
            
            if font_large:
                bbox = draw.textbbox((0, 0), line, font=font_large)
                text_width = bbox[2] - bbox[0]
            else:
                text_width = len(line) * 10
            
            x_pos = (width - text_width) // 2
            
            draw.text((x_pos, y_pos), line, font=font_large, fill=(40, 40, 40))
        
        # Add decorative border
        draw.rectangle([(margin-10, margin-10), (width-margin+10, height-margin+10)], 
                      outline=(200, 180, 160), width=3)
        
        # Add footer
        footer_text = "God AI 🌸"
        if font_small:
            bbox = draw.textbbox((0, 0), footer_text, font=font_small)
            footer_width = bbox[2] - bbox[0]
        else:
            footer_width = len(footer_text) * 8
        
        draw.text((width - footer_width - margin, height - 40), footer_text, 
                 font=font_small, fill=(120, 120, 120))
        
        # Save image
        image_filename = f"{verse_id.replace(' ', '_').replace('.', '_')}.png"
        image_path = os.path.join(MEDIA_ROOT, "images", image_filename)
        img.save(image_path, "PNG")
        
        return f"/media/images/{image_filename}"
    except Exception as e:
        print(f"Image generation failed: {e}")
        return None

# ----------------------
# RAG INTEGRATION
# ----------------------

def get_relevant_verse(user_message: str, mood: Optional[str] = None, user_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Get relevant verse based on user message, mood, and avoid recent verses using RAG
    """
    # Get RAG instance
    rag = get_rag_instance()
    if not rag:
        raise HTTPException(status_code=500, detail="RAG system not initialized. Please ensure embeddings are generated and RAG is configured correctly.")
    
    # Get user's recent verses to avoid repetition
    recent_verses = set()
    if user_id:
        with Session(engine) as session:
            user = session.get(User, user_id)
            if user and user.recent_verses:
                recent_data = json.loads(user.recent_verses)
                for mood_key, verses in recent_data.items():
                    recent_verses.update(verses)
    
    try:
        # Use RAG for intelligent verse selection
        relevant_verses = rag.find_relevant_verses(
            query=user_message,
            mood=mood,
            user_id=user_id,
            recent_verses=recent_verses,
            top_k=1
        )
    except Exception as e:
        print(f"RAG retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="RAG retrieval failed. Please check RAG configuration and embeddings.")

    if not relevant_verses:
        raise HTTPException(status_code=500, detail="No relevant verses found by RAG.")

    verse = relevant_verses[0]
    # Truncate very long verses to prevent overwhelming responses
    text = verse["text"]
    if len(text) > 500:
        # Try to find a good breaking point (end of sentence)
        truncated = text[:500]
        last_period = truncated.rfind('.')
        if last_period > 200:  # If we can find a period in the last 300 chars
            text = truncated[:last_period + 1] + "..."
        else:
            text = truncated + "..."

    return {
        "verse_id": verse["verse_id"],
        "text": text,
        "source": verse["source"]
    }

def generate_response(user_message: str, verse: Dict[str, Any], mood: str, user_summary: Optional[str] = None) -> str:
    """
    Generate AI response combining verse and context using LLM or enhanced template
    """
    llm = get_llm_service()
    if not llm:
        raise HTTPException(status_code=500, detail="LLM service not initialized. Please configure Ollama or Groq.")

    try:
        return llm.generate_response(user_message, verse, mood, user_summary)
    except Exception as e:
        print(f"LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail="LLM generation failed. Please check LLM configuration.")

# ----------------------
# FASTAPI APP
# ----------------------

app = FastAPI(
    title="God AI Backend",
    description="A spiritual AI companion with mood detection and verse recommendations",
    version="1.0.0"
)

# Initialize RAG system on startup
@app.on_event("startup")
def startup_event():
    """Initialize RAG system and other startup tasks"""
    print("Initializing God AI Backend...")
    try:
        initialize_rag(engine)
        print("✓ RAG system initialized")
    except Exception as e:
        print(f"Warning: RAG initialization failed: {e}")
    
    try:
        initialize_llm()
        print("✓ LLM service initialized")
    except Exception as e:
        print(f"Warning: LLM initialization failed: {e}")
    
    print("✓ Backend ready!")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# PYDANTIC MODELS
# ----------------------

class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    name: str

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    detected_mood: str
    verse_id: str
    verse_text: str
    verse_source: str

class SaveVerseRequest(BaseModel):
    verse_id: str

class DailyVerseResponse(BaseModel):
    verse_id: str
    text: str
    source: str
    audio_url: Optional[str] = None
    image_url: Optional[str] = None

class UserProfile(BaseModel):
    user_id: str
    name: str
    email: str
    last_mood: Optional[str]
    recent_verses: Dict[str, List[str]]
    saved_verses: List[str]
    chat_history: List[Dict[str, Any]]

# ----------------------
# AUTHENTICATION
# ----------------------

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_uuid = payload.get("sub")
    
    with Session(engine) as session:
        stmt = select(User).where(User.uuid == user_uuid)
        user = session.exec(stmt).one_or_none()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user

# ----------------------
# AUTH ROUTES
# ----------------------

@app.post("/signup", response_model=TokenResponse)
def signup(request: SignupRequest):
    with Session(engine) as session:
        # Check if email already exists
        existing_user = session.exec(select(User).where(User.email == request.email)).one_or_none()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create new user
        hashed_password = hash_password(request.password)
        user = User(
            name=request.name,
            email=request.email,
            hashed_password=hashed_password
        )
        
        session.add(user)
        session.commit()
        session.refresh(user)
        
        # Create token
        token = create_access_token({"sub": user.uuid})
        
        return TokenResponse(
            access_token=token,
            user_id=user.uuid,
            name=user.name
        )

@app.post("/login", response_model=TokenResponse)
def login(request: LoginRequest):
    with Session(engine) as session:
        user = session.exec(select(User).where(User.email == request.email)).one_or_none()
        
        if not user or not verify_password(request.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Incorrect email or password")
        
        token = create_access_token({"sub": user.uuid})
        
        return TokenResponse(
            access_token=token,
            user_id=user.uuid,
            name=user.name
        )

# ----------------------
# CHAT ROUTE
# ----------------------

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, user: User = Depends(get_current_user)):
    message = request.message
    
    # 1. Mood detection
    mood = "neutral"
    if mood_pipeline:
        try:
            mood_results = mood_pipeline(message)
            # Handle nested list structure from transformers pipeline
            if isinstance(mood_results, list) and len(mood_results) > 0:
                # Extract the actual results from the nested structure
                actual_results = mood_results[0] if isinstance(mood_results[0], list) else mood_results
                if len(actual_results) > 0:
                    top_result = max(actual_results, key=lambda x: x['score'])
                    mood = top_result['label'].lower()
        except Exception as e:
            print(f"Mood detection failed: {e}")
    
    # 2. Store mood and usage log
    with Session(engine) as session:
        db_user = session.get(User, user.id)
        db_user.last_mood = mood
        db_user.updated_at = datetime.utcnow()
        
        session.add(UsageLog(user_id=user.id, mood=mood, endpoint="chat"))
        session.commit()
    
    # 3. Summarization
    summary = message[:200]  # Fallback
    if summary_pipeline:
        try:
            # Calculate appropriate max_length based on input length
            # Use a reasonable ratio: max_length should be at least 10, but not more than input length
            input_length = len(message.split())
            max_len = max(10, min(60, max(10, input_length // 2)))  # At least 10, at most 60, or half of input
            min_len = max(5, min(10, max_len // 2))  # At least 5, at most 10, or half of max_len
            
            summary_result = summary_pipeline(message, max_length=max_len, min_length=min_len)
            if isinstance(summary_result, list) and len(summary_result) > 0:
                summary = summary_result[0]['summary_text']
        except Exception as e:
            print(f"Summarization failed: {e}")
    
    # 4. Get relevant verse
    verse = get_relevant_verse(message, mood=mood, user_id=user.id)
    
    # 5. Generate response
    reply = generate_response(message, verse, mood, summary)
    
    # 6. Store chat summary and update recent verses
    with Session(engine) as session:
        # Add chat summary
        session.add(ChatSummary(
            user_id=user.id,
            mood=mood,
            summary=summary,
            verse_id=verse['verse_id']
        ))
        
        # Update recent verses
        db_user = session.get(User, user.id)
        recent_verses = json.loads(db_user.recent_verses or "{}")
        
        if mood not in recent_verses:
            recent_verses[mood] = []
        
        # Keep only last 3 verses per mood
        recent_verses[mood].append(verse['verse_id'])
        if len(recent_verses[mood]) > 3:
            recent_verses[mood] = recent_verses[mood][-3:]
        
        db_user.recent_verses = json.dumps(recent_verses)
        session.commit()
    
    return ChatResponse(
        reply=reply,
        detected_mood=mood,
        verse_id=verse['verse_id'],
        verse_text=verse['text'],
        verse_source=verse['source']
    )

# ----------------------
# DAILY VERSE ROUTE
# ----------------------

@app.get("/daily-verse", response_model=DailyVerseResponse)
def daily_verse(user: User = Depends(get_current_user)):
    """Get today's personalized daily verse with audio and image"""
    # Get verse based on user's last mood
    mood = user.last_mood or "neutral"
    verse = get_relevant_verse("daily verse", mood=mood, user_id=user.id)
    
    # Generate audio and image if they don't exist
    audio_filename = f"{verse['verse_id'].replace(' ', '_').replace('.', '_')}.mp3"
    image_filename = f"{verse['verse_id'].replace(' ', '_').replace('.', '_')}.png"
    
    audio_path = os.path.join(MEDIA_ROOT, "audio", audio_filename)
    image_path = os.path.join(MEDIA_ROOT, "images", image_filename)
    
    # Only generate if files don't exist (for performance)
    audio_url = None
    image_url = None
    
    if not os.path.exists(audio_path):
        audio_url = generate_tts(verse['text'], verse['verse_id'])
    else:
        audio_url = f"/media/audio/{audio_filename}"
    
    if not os.path.exists(image_path):
        image_url = generate_verse_image(verse['text'], verse['verse_id'])
    else:
        image_url = f"/media/images/{image_filename}"
    
    # Log daily verse access
    with Session(engine) as session:
        session.add(UsageLog(user_id=user.id, mood=mood, endpoint="daily-verse"))
        session.commit()
    
    return DailyVerseResponse(
        verse_id=verse['verse_id'],
        text=verse['text'],
        source=verse['source'],
        audio_url=audio_url,
        image_url=image_url
    )

# ----------------------
# SAVE VERSE ROUTES
# ----------------------

@app.post("/save-verse")
def save_verse(request: SaveVerseRequest, user: User = Depends(get_current_user)):
    """Save a verse to user's favorites"""
    with Session(engine) as session:
        db_user = session.get(User, user.id)
        saved_verses = json.loads(db_user.saved_verses or "[]")
        
        if request.verse_id not in saved_verses:
            saved_verses.append(request.verse_id)
            db_user.saved_verses = json.dumps(saved_verses)
            session.commit()
            
            return {"success": True, "message": "Verse saved successfully"}
        else:
            return {"success": False, "message": "Verse already saved"}

@app.post("/save-verse-from-daily")
def save_verse_from_daily(user: User = Depends(get_current_user)):
    """Automatically save the current daily verse to user's favorites"""
    # Get the current daily verse
    mood = user.last_mood or "neutral"
    verse = get_relevant_verse("daily verse", mood=mood, user_id=user.id)
    
    with Session(engine) as session:
        db_user = session.get(User, user.id)
        saved_verses = json.loads(db_user.saved_verses or "[]")
        
        if verse['verse_id'] not in saved_verses:
            saved_verses.append(verse['verse_id'])
            db_user.saved_verses = json.dumps(saved_verses)
            session.commit()
            
            return {
                "success": True, 
                "message": "Daily verse saved successfully",
                "verse_id": verse['verse_id'],
                "verse_text": verse['text'],
                "source": verse['source']
            }
        else:
            return {
                "success": False, 
                "message": "Daily verse already saved",
                "verse_id": verse['verse_id']
            }

@app.get("/daily-verse-with-save")
def daily_verse_with_save(user: User = Depends(get_current_user)):
    """Get daily verse with save status (no automatic saving)"""
    # Get the daily verse
    mood = user.last_mood or "neutral"
    verse = get_relevant_verse("daily verse", mood=mood, user_id=user.id)
    
    # Generate audio and image if they don't exist
    audio_filename = f"{verse['verse_id'].replace(' ', '_').replace('.', '_')}.mp3"
    image_filename = f"{verse['verse_id'].replace(' ', '_').replace('.', '_')}.png"
    
    audio_path = os.path.join(MEDIA_ROOT, "audio", audio_filename)
    image_path = os.path.join(MEDIA_ROOT, "images", image_filename)
    
    audio_url = None
    image_url = None
    
    if not os.path.exists(audio_path):
        audio_url = generate_tts(verse['text'], verse['verse_id'])
    else:
        audio_url = f"/media/audio/{audio_filename}"
    
    if not os.path.exists(image_path):
        image_url = generate_verse_image(verse['text'], verse['verse_id'])
    else:
        image_url = f"/media/images/{image_filename}"
    
    # Check if verse is already saved (but don't auto-save)
    with Session(engine) as session:
        db_user = session.get(User, user.id)
        saved_verses = json.loads(db_user.saved_verses or "[]")
        is_already_saved = verse['verse_id'] in saved_verses
        
        # Log daily verse access
        session.add(UsageLog(user_id=user.id, mood=mood, endpoint="daily-verse"))
        session.commit()
        
        return {
            "verse_id": verse['verse_id'],
            "text": verse['text'],
            "source": verse['source'],
            "audio_url": audio_url,
            "image_url": image_url,
            "is_saved": is_already_saved,
            "message": "Daily verse loaded successfully"
        }

@app.get("/my-saved-verses")
def my_saved_verses(user: User = Depends(get_current_user)):
    """Get all saved verses for the user"""
    with Session(engine) as session:
        db_user = session.get(User, user.id)
        saved_verse_ids = json.loads(db_user.saved_verses or "[]")
        
        verses = []
        for verse_id in saved_verse_ids:
            verse = session.exec(select(Verse).where(Verse.verse_id == verse_id)).one_or_none()
            if verse:
                verses.append({
                    "verse_id": verse.verse_id,
                    "text": verse.text,
                    "source": verse.source,
                    "image_url": f"/media/images/{verse_id.replace(' ', '_').replace('.', '_')}.png",
                    "audio_url": f"/media/audio/{verse_id.replace(' ', '_').replace('.', '_')}.mp3"
                })
        
        return {"saved_verses": verses}

# ----------------------
# HISTORY ROUTES
# ----------------------

@app.get("/history/{user_id}")
def get_history(user_id: int, current_user: User = Depends(get_current_user)):
    # Users can only see their own history unless they're admin
    if not current_user.is_admin and current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    with Session(engine) as session:
        chat_summaries = session.exec(
            select(ChatSummary)
            .where(ChatSummary.user_id == user_id)
            .order_by(ChatSummary.date.desc())
            .limit(50)
        ).all()
        
        history = []
        for summary in chat_summaries:
            history.append({
                "date": summary.date.isoformat(),
                "mood": summary.mood,
                "summary": summary.summary,
                "verse_id": summary.verse_id
            })
        
        return {"history": history}

@app.get("/last-session")
def last_session(user: User = Depends(get_current_user)):
    with Session(engine) as session:
        last_summary = session.exec(
            select(ChatSummary)
            .where(ChatSummary.user_id == user.id)
            .order_by(ChatSummary.date.desc())
        ).first()
        
        if not last_summary:
            return {"message": "No previous session found."}
        
        return {
            "message": f"Last time on {last_summary.date.date().isoformat()} you were feeling {last_summary.mood}. How are you feeling now?",
            "last_session": {
                "date": last_summary.date.isoformat(),
                "mood": last_summary.mood,
                "summary": last_summary.summary,
                "verse_id": last_summary.verse_id
            }
        }

# ----------------------
# USER PROFILE ROUTE
# ----------------------

@app.get("/profile", response_model=UserProfile)
def get_profile(user: User = Depends(get_current_user)):
    with Session(engine) as session:
        # Get recent chat summaries
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

@app.get("/chats/recent")
def get_recent_chats(user: User = Depends(get_current_user)):
    """Get recent chat summaries for sidebar"""
    with Session(engine) as session:
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

@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: int, user: User = Depends(get_current_user)):
    """Delete a chat summary"""
    with Session(engine) as session:
        # Refresh user from database to ensure we have the latest data
        db_user = session.get(User, user.id)
        if not db_user:
            raise HTTPException(status_code=401, detail="User not found")
        
        chat = session.get(ChatSummary, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Ensure user can only delete their own chats
        # Use the same filtering logic as get_recent_chats to ensure consistency
        # Check if this chat belongs to the current user
        user_chat = session.exec(
            select(ChatSummary)
            .where(ChatSummary.id == chat_id)
            .where(ChatSummary.user_id == db_user.id)
        ).one_or_none()
        
        if not user_chat:
            print(f"DEBUG: Delete chat authorization failed - chat.user_id={chat.user_id}, db_user.id={db_user.id}, chat_id={chat_id}, user.email={db_user.email}")
            raise HTTPException(
                status_code=403, 
                detail="Not authorized to delete this chat. This chat belongs to a different user."
            )
        
        session.delete(user_chat)
        session.commit()
        
        return {"success": True, "message": "Chat deleted"}

# ----------------------
# ADMIN ROUTES
# ----------------------

def require_admin_user(user: User):
    if not user.is_admin:
        raise HTTPException(
            status_code=403, 
            detail=f"Admin access required. Current user '{user.email}' is not an admin."
        )

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
    """Create demo flagged comments if table is empty to showcase UI."""
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

@app.get("/admin/analytics")
def admin_analytics(current_user: User = Depends(get_current_user)):
    require_admin_user(current_user)
    
    with Session(engine) as session:
        # Active users (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        active_users = session.exec(
            select(UsageLog.user_id)
            .where(UsageLog.timestamp >= thirty_days_ago)
        ).all()
        active_user_count = len(set(active_users))
        
        # Mood distribution (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        mood_logs = session.exec(
            select(UsageLog)
            .where(UsageLog.timestamp >= week_ago)
        ).all()
        
        mood_counts = {}
        for log in mood_logs:
            if log.mood:
                mood_counts[log.mood] = mood_counts.get(log.mood, 0) + 1
        
        # Peak usage hours (last 7 days)
        hour_counts = {}
        for log in mood_logs:
            hour = log.timestamp.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        peak_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Most popular verses
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
            "total_users": session.exec(select(User)).all().__len__(),
            "total_verses": session.exec(select(Verse)).all().__len__()
        }

@app.get("/admin/system-health")
def admin_system_health(current_user: User = Depends(get_current_user)):
    require_admin_user(current_user)
    
    db_status = True
    db_latency_ms = None
    try:
        start = datetime.utcnow()
        with Session(engine) as session:
            session.exec(select(User).limit(1)).first()
        end = datetime.utcnow()
        db_latency_ms = int((end - start).total_seconds() * 1000)
    except Exception as e:
        print(f"Database health check failed: {e}")
        db_status = False
    
    llm = get_llm_service()
    rag = get_rag_instance()
    if llm:
        provider_attr = getattr(llm, "provider", None)
        if isinstance(provider_attr, str):
            llm_provider = provider_attr
        elif provider_attr:
            llm_provider = provider_attr.__class__.__name__
        else:
            llm_provider = llm.__class__.__name__
    else:
        llm_provider = "unavailable"
    
    return {
        "application": True,
        "database": db_status,
        "api": True,
        "llm": llm is not None,
        "details": {
            "database_latency_ms": db_latency_ms,
            "rag_loaded": bool(rag),
            "llm_provider": llm_provider,
            "timestamp": datetime.utcnow().isoformat()
        }
    }

@app.get("/admin/recent-activity")
def admin_recent_activity(current_user: User = Depends(get_current_user)):
    require_admin_user(current_user)
    
    with Session(engine) as session:
        logs = session.exec(
            select(UsageLog).order_by(UsageLog.timestamp.desc()).limit(20)
        ).all()
        
        user_cache: Dict[int, User] = {}
        activity = []
        for log in logs:
            if log.user_id not in user_cache:
                user_cache[log.user_id] = session.get(User, log.user_id)
            user = user_cache.get(log.user_id)
            user_name = user.name if user else "Unknown user"
            
            if log.endpoint in ("chat", "daily-verse"):
                item_type = "chat"
                detail = f"{user_name} engaged with the AI ({log.endpoint})."
            elif log.endpoint in ("save-verse", "save-verse-from-daily", "daily-verse-with-save"):
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

@app.get("/admin/analytics/engagement")
def admin_engagement_metrics(
    days: int = Query(30, ge=1, le=90),
    current_user: User = Depends(get_current_user)
):
    require_admin_user(current_user)
    
    since = datetime.utcnow() - timedelta(days=days)
    with Session(engine) as session:
        logs = session.exec(
            select(UsageLog).where(UsageLog.timestamp >= since)
        ).all()
    
    event_counts: Dict[str, int] = defaultdict(int)
    daily_active_users: Dict[str, set] = defaultdict(set)
    
    for log in logs:
        event_name = (log.endpoint or "unknown").replace("/", "_")
        event_counts[event_name] += 1
        if log.user_id:
            day_key = log.timestamp.date().isoformat()
            daily_active_users[day_key].add(log.user_id)
    
    dau_counts = {day: len(users) for day, users in daily_active_users.items()}
    total_events = sum(event_counts.values())
    
    return {
        "event_counts": event_counts,
        "daily_active_users": dau_counts,
        "total_events": total_events
    }

@app.get("/admin/analytics/verse-popularity")
def admin_verse_popularity(
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user)
):
    require_admin_user(current_user)
    
    with Session(engine) as session:
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
            verse_records = session.exec(
                select(Verse).where(Verse.verse_id.in_(verse_ids))
            ).all()
            verses = {v.verse_id: v for v in verse_records}
        
        response = []
        for verse_id, views in rows:
            if not verse_id:
                continue
            verse = verses.get(verse_id)
            response.append({
                "verse_id": verse_id,
                "views": views,
                "text": verse.text if verse else "",
                "source": verse.source if verse else "Unknown"
            })
        
        return response

@app.get("/admin/moderation/flagged")
def admin_flagged_comments(current_user: User = Depends(get_current_user)):
    require_admin_user(current_user)
    
    with Session(engine) as session:
        ensure_sample_flagged_comments(session)
        comments = session.exec(
            select(FlaggedComment)
            .where(FlaggedComment.status == "pending")
            .order_by(FlaggedComment.created_at.desc())
        ).all()
        return [serialize_flagged_comment(comment) for comment in comments]

@app.post("/admin/moderation/{comment_id}/approve")
def admin_approve_comment(comment_id: int, current_user: User = Depends(get_current_user)):
    require_admin_user(current_user)
    
    with Session(engine) as session:
        comment = session.get(FlaggedComment, comment_id)
        if not comment:
            raise HTTPException(status_code=404, detail="Flagged comment not found")
        comment.status = "approved"
        session.add(comment)
        session.commit()
        session.refresh(comment)
        return {
            "success": True,
            "message": "Comment approved",
            "comment": serialize_flagged_comment(comment)
        }

@app.delete("/admin/moderation/{comment_id}/delete")
def admin_delete_comment(comment_id: int, current_user: User = Depends(get_current_user)):
    require_admin_user(current_user)
    
    with Session(engine) as session:
        comment = session.get(FlaggedComment, comment_id)
        if not comment:
            raise HTTPException(status_code=404, detail="Flagged comment not found")
        session.delete(comment)
        session.commit()
        return {"success": True, "message": "Comment deleted"}

@app.get("/admin/users")
def admin_users(
    search: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user)
):
    """Get all users (admin only)"""
    require_admin_user(current_user)
    
    with Session(engine) as session:
        stmt = select(User)
        
        # Apply search filter if provided
        if search:
            search_term = f"%{search}%"
            # SQLite uses LIKE (case-insensitive by default), PostgreSQL uses ILIKE
            try:
                stmt = stmt.where(
                    (User.name.ilike(search_term)) | (User.email.ilike(search_term))
                )
            except:
                # Fallback for SQLite
                stmt = stmt.where(
                    (User.name.like(search_term)) | (User.email.like(search_term))
                )
        
        users = session.exec(stmt.order_by(User.created_at.desc())).all()
        
        return [
            {
                "id": user.id,
                "uuid": user.uuid,
                "name": user.name,
                "email": user.email,
                "is_admin": user.is_admin,
                "created_at": user.created_at.isoformat(),
                "last_mood": user.last_mood
            }
            for user in users
        ]

class UserUpdateRequest(BaseModel):
    is_admin: Optional[bool] = None
    name: Optional[str] = None

@app.put("/admin/users/{user_id}")
def admin_update_user(
    user_id: int,
    update_data: UserUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    """Update user (admin only)"""
    require_admin_user(current_user)
    
    with Session(engine) as session:
        user = session.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Prevent demoting yourself
        if current_user.id == user_id and update_data.get("is_admin") is False:
            raise HTTPException(status_code=400, detail="Cannot demote yourself")
        
        # Update fields
        if "is_admin" in update_data:
            user.is_admin = update_data["is_admin"]
        
        if "name" in update_data:
            user.name = update_data["name"]
        
        user.updated_at = datetime.utcnow()
        session.add(user)
        session.commit()
        session.refresh(user)
        
        return {
            "id": user.id,
            "uuid": user.uuid,
            "name": user.name,
            "email": user.email,
            "is_admin": user.is_admin,
            "created_at": user.created_at.isoformat(),
            "last_mood": user.last_mood
        }

@app.delete("/admin/users/{user_id}")
def admin_delete_user(
    user_id: int,
    current_user: User = Depends(get_current_user)
):
    """Delete user (admin only)"""
    require_admin_user(current_user)
    
    with Session(engine) as session:
        user = session.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Prevent deleting yourself
        if current_user.id == user_id:
            raise HTTPException(status_code=400, detail="Cannot delete yourself")
        
        # Prevent deleting other admins
        if user.is_admin:
            raise HTTPException(status_code=400, detail="Cannot delete admin users")
        
        session.delete(user)
        session.commit()
        
        return {"success": True, "message": "User deleted"}

# ----------------------
# PDF PROCESSING ROUTE
# ----------------------

@app.post("/admin/process-pdfs")
def process_pdfs(current_user: User = Depends(get_current_user)):
    """Process PDF files and extract verses (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        from pdf_processor import process_all_pdfs, save_verses_to_database
        
        # Process all PDFs
        verses_dict = process_all_pdfs()
        
        # Save to database
        save_verses_to_database(verses_dict)
        
        # Reinitialize RAG system with new verses
        try:
            initialize_rag(engine)
            print("✓ RAG system updated with new verses")
        except Exception as e:
            print(f"Warning: Could not update RAG system: {e}")
        
        # Calculate totals
        total_verses = sum(len(verses) for verses in verses_dict.values())
        
        return {
            "success": True,
            "message": f"Successfully processed {len(verses_dict)} PDF files",
            "total_verses_added": total_verses,
            "files_processed": list(verses_dict.keys()),
            "verses_per_file": {k: len(v) for k, v in verses_dict.items()}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

@app.get("/admin/llm-status")
def llm_status(current_user: User = Depends(get_current_user)):
    """Get LLM service status (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        llm = get_llm_service()
        if not llm:
            return {"error": "LLM service not initialized"}
        
        return llm.get_model_info()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting LLM status: {str(e)}")

@app.post("/admin/embedding-status")
def embedding_status(current_user: User = Depends(get_current_user)):
    """Get embedding storage status (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        rag = get_rag_instance()
        if not rag:
            return {"error": "RAG system not initialized"}
        
        storage_info = rag.get_storage_info()
        
        # Add current memory info
        storage_info.update({
            "current_embeddings_in_memory": len(rag.verse_embeddings),
            "current_metadata_in_memory": len(rag.verse_metadata)
        })
        
        return storage_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting embedding status: {str(e)}")

@app.post("/admin/regenerate-embeddings")
def regenerate_embeddings(current_user: User = Depends(get_current_user)):
    """Force regeneration of all embeddings (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        rag = get_rag_instance()
        if not rag:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        rag.regenerate_embeddings()
        
        return {
            "success": True,
            "message": "Embeddings regenerated successfully",
            "total_embeddings": len(rag.verse_embeddings)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error regenerating embeddings: {str(e)}")

# ----------------------
# SEED DATA ROUTE
# ----------------------

@app.post("/admin/seed-verses")
def seed_verses(current_user: User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    sample_verses = [
        {
            "verse_id": "Gita_2.47",
            "text": "You have a right to perform your prescribed duties, but not to the fruits of your actions. Never consider yourself the cause of the results of your activities, and never be attached to not doing your duty.",
            "source": "Bhagavad Gita",
            "mood_tags": "[\"neutral\", \"purpose\", \"duty\"]"
        },
        {
            "verse_id": "Psalm_23.1",
            "text": "The Lord is my shepherd; I shall not want. He maketh me to lie down in green pastures: he leadeth me beside the still waters.",
            "source": "Bible - Psalms",
            "mood_tags": "[\"comfort\", \"peace\", \"sad\"]"
        },
        {
            "verse_id": "Quran_94.5",
            "text": "So, surely with hardship comes ease. Surely with hardship comes ease.",
            "source": "Quran",
            "mood_tags": "[\"hope\", \"difficulty\", \"sad\"]"
        },
        {
            "verse_id": "Matthew_11.28",
            "text": "Come unto me, all ye that labour and are heavy laden, and I will give you rest.",
            "source": "Bible - Matthew",
            "mood_tags": "[\"comfort\", \"rest\", \"tired\"]"
        },
        {
            "verse_id": "Gita_4.8",
            "text": "To protect the righteous, to annihilate the miscreants, and to reestablish the principles of dharma, I advent Myself millennium after millennium.",
            "source": "Bhagavad Gita",
            "mood_tags": "[\"hope\", \"justice\", \"purpose\"]"
        },
        {
            "verse_id": "Psalm_46.1",
            "text": "God is our refuge and strength, a very present help in trouble.",
            "source": "Bible - Psalms",
            "mood_tags": "[\"strength\", \"comfort\", \"fear\"]"
        },
        {
            "verse_id": "Quran_13.28",
            "text": "Those who believe, and whose hearts find satisfaction in the remembrance of Allah: for without doubt in the remembrance of Allah do hearts find satisfaction.",
            "source": "Quran",
            "mood_tags": "[\"peace\", \"remembrance\", \"faith\"]"
        },
        {
            "verse_id": "Proverbs_3.5",
            "text": "Trust in the Lord with all thine heart; and lean not unto thine own understanding.",
            "source": "Bible - Proverbs",
            "mood_tags": "[\"trust\", \"wisdom\", \"guidance\"]"
        }
    ]
    
    with Session(engine) as session:
        added_count = 0
        for verse_data in sample_verses:
            existing = session.exec(
                select(Verse).where(Verse.verse_id == verse_data["verse_id"])
            ).one_or_none()
            
            if not existing:
                verse = Verse(**verse_data)
                session.add(verse)
                added_count += 1
        
        session.commit()
    
    return {"success": True, "verses_added": added_count}

# ----------------------
# STATIC FILES
# ----------------------

from fastapi.staticfiles import StaticFiles
app.mount("/media", StaticFiles(directory=MEDIA_ROOT), name="media")

# ----------------------
# HEALTH CHECK
# ----------------------

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/stats/public")
def public_stats():
    """Get public statistics for landing page (no authentication required)"""
    with Session(engine) as session:
        # Total users
        total_users = session.exec(select(func.count(User.id))).one()
        
        # Active users (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        active_user_ids = session.exec(
            select(UsageLog.user_id)
            .where(UsageLog.timestamp >= thirty_days_ago)
        ).all()
        active_users = len(set(active_user_ids)) if active_user_ids else 0
        
        # Total conversations (chat summaries)
        total_conversations = session.exec(select(func.count(ChatSummary.id))).one()
        
        # Verses shared (unique verses in chat summaries)
        verses_shared = session.exec(
            select(func.count(func.distinct(ChatSummary.verse_id)))
            .where(ChatSummary.verse_id.is_not(None))
        ).one()
        
        return {
            "active_users": active_users if active_users > 0 else total_users,
            "verses_shared": verses_shared,
            "conversations": total_conversations
        }

@app.get("/admin/check")
def admin_check(user: User = Depends(get_current_user)):
    """Check if current user is admin - useful for debugging"""
    return {
        "is_admin": user.is_admin,
        "user_id": user.uuid,
        "email": user.email,
        "name": user.name
    }

# ----------------------
# RUNNING THE APP
# ----------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
