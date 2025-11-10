from datetime import datetime
from typing import Optional
import uuid

from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()), index=True)
    name: str
    email: str = Field(index=True, unique=True)
    hashed_password: str
    is_admin: bool = Field(default=False)
    last_mood: Optional[str] = None
    recent_verses: Optional[str] = Field(default="{}")  # JSON object with mood -> verse_ids
    saved_verses: Optional[str] = Field(default="[]")  # JSON list of verse_ids
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ChatSummary(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    date: datetime = Field(default_factory=datetime.utcnow)
    mood: Optional[str] = None
    summary: str
    verse_id: Optional[str] = None


class UsageLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    mood: Optional[str] = None
    endpoint: Optional[str] = None


class FlaggedComment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    user_name: str
    user_email: str
    verse_id: str
    comment: str
    status: str = Field(default="pending", index=True)
    reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Verse(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    verse_id: str = Field(index=True, unique=True)  # e.g., "Gita_2.47"
    text: str
    source: str  # e.g., "Bhagavad Gita"
    mood_tags: Optional[str] = "[]"  # JSON list, e.g., ["sad","hopeful"]
    created_at: datetime = Field(default_factory=datetime.utcnow)
