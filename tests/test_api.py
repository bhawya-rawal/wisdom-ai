import pytest
from fastapi.testclient import TestClient
from config.settings import settings

# Force SQLite fallback for fast local testing
settings.USE_SQLITE_FALLBACK = True
settings.DATABASE_URL = "sqlite:///./test_db.db"

from main import app
from database.connection import init_db

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    """Initializes tables in the temporary sqlite testing database"""
    init_db()

def test_health_endpoint():
    """Verify that the health check endpoint is active and returns healthy status"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_public_stats_endpoint():
    """Verify public stats statistics route"""
    response = client.get("/stats/public")
    assert response.status_code == 200
    data = response.json()
    assert "active_users" in data
    assert "conversations" in data
    assert "verses_shared" in data

def test_signup_validation():
    """Verify that user signup validates schemas and enforces emails format"""
    # Send incorrect email format
    response = client.post("/signup", json={
        "name": "Test User",
        "email": "not-an-email",
        "password": "password123"
    })
    assert response.status_code == 422 # Unprocessable Entity
