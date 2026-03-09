import hashlib
import jwt
from datetime import datetime, timedelta
from typing import Optional
from config.settings import settings

class AuthService:
    """Manages secure password hashing and JWT token processing"""
    
    def __init__(self):
        self.secret = settings.JWT_SECRET
        self.algorithm = settings.JWT_ALGORITHM
        self.expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES

    def hash_password(self, password: str) -> str:
        """Hash a raw password with SHA-256 (matches original implementation)"""
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify standard password match"""
        return self.hash_password(password) == hashed

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Generate JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.expire_minutes)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret, algorithm=self.algorithm)

    def decode_token(self, token: str) -> dict:
        """Decode and validate a JWT access token"""
        try:
            return jwt.decode(token, self.secret, algorithms=[self.algorithm])
        except Exception:
            raise ValueError("Invalid token")

# Global auth service instance
auth_service = AuthService()
