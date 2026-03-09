from typing import Optional, Dict, List, Any
from pydantic import BaseModel, EmailStr

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

# Structured citation response
class Citation(BaseModel):
    source: str       # e.g., "Bhagavad Gita" or "Bible"
    location: str     # e.g., "Chapter 2 Verse 47" or "John 3:16"
    text: str         # The verse text

class ChatResponse(BaseModel):
    reply: str
    detected_mood: str
    verse_id: str
    verse_text: str
    verse_source: str
    
    # Extended production-grade RAG fields
    citations: Optional[List[Citation]] = None
    confidence: Optional[float] = None          # 0 to 100
    confidence_label: Optional[str] = None      # "High", "Medium", "Low"
    conversation_summary: Optional[str] = None  # Automated summary of the window
    latency_ms: Optional[Dict[str, float]] = None # Observability track breakdown

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
