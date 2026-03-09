import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    # App Settings
    APP_TITLE: str = "WisdomAI Backend"
    APP_DESCRIPTION: str = "A production-grade spiritual AI companion with mood detection and FAISS-based verse recommendations"
    APP_VERSION: str = "2.0.0"
    
    # Security
    JWT_SECRET: str = os.getenv("JWT_SECRET", "your-super-secret-key-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # Admin Credentials
    ADMIN_EMAIL: str = os.getenv("ADMIN_EMAIL", "admin@godai.com")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin123")
    ADMIN_NAME: str = os.getenv("ADMIN_NAME", "God AI Admin")
    
    # Media & Directories
    MEDIA_ROOT: str = os.getenv("MEDIA_ROOT", str(BASE_DIR / "media"))
    
    # Database
    # Default to local PostgreSQL container if no DATABASE_URL is set (Neon in production)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:postgres@localhost:5432/wisdom_ai"
    )
    # Fix postgres prefix for sqlalchemy
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        
    # Local fallback option if database fails or local developer testing without docker postgres
    USE_SQLITE_FALLBACK: bool = os.getenv("USE_SQLITE_FALLBACK", "false").lower() == "true"
    SQLITE_URL: str = "sqlite:///./god_ai.db"
    
    # LLM Services
    USE_GROQ: bool = os.getenv("USE_GROQ", "true").lower() == "true"
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_API_URL: str = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    
    USE_OLLAMA: bool = os.getenv("USE_OLLAMA", "false").lower() == "true"
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "tinyllama")
    
    # Prompt Versioning
    PROMPT_VERSION: str = os.getenv("PROMPT_VERSION", "v1")  # Supported: v1, v2, v3
    
    # Retrieval & Embeddings
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    FAISS_INDEX_DIR: str = os.getenv("FAISS_INDEX_DIR", str(BASE_DIR / "embeddings"))
    
    # Reranking
    ENABLE_RERANKING: bool = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
    RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # RAG Tuning & Guardrails
    RAG_TOP_K_FAISS: int = int(os.getenv("RAG_TOP_K_FAISS", "20"))
    RAG_TOP_K_FINAL: int = int(os.getenv("RAG_TOP_K_FINAL", "5"))
    CONFIDENCE_THRESHOLD_HIGH: float = 85.0
    CONFIDENCE_THRESHOLD_MEDIUM: float = 70.0
    
    # Safety Check / Hallucination fallback
    ENABLE_GUARDRAILS: bool = os.getenv("ENABLE_GUARDRAILS", "true").lower() == "true"
    FALLBACK_RESPONSE: str = "I couldn't find reliable supporting passages for this question in our scriptural database."
    
    # Observability
    ENABLE_PROMETHEUS: bool = os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true"
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", "9090"))
    
    # RAGAS Evaluation
    RAGAS_JUDGE_MODEL: str = os.getenv("RAGAS_JUDGE_MODEL", "llama-3.1-8b-instant")
    
    # MCP (Model Context Protocol) Config
    ENABLE_MCP: bool = os.getenv("ENABLE_MCP", "false").lower() == "true"

settings = Settings()
