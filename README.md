God AI Backend (FastAPI) – Setup and Usage

Overview
God AI is a FastAPI backend that serves as a spiritual companion. It detects mood, recommends verses across scriptures (Bhagavad Gita, Bible, Quran), and generates multimedia (TTS audio and shareable images). It optionally integrates with an LLM backend (local Transformers or Ollama, with optional Grok API) and uses a lightweight RAG layer with prebuilt embeddings.

Key Features
- Mood detection and summarization (Transformers pipelines)
- RAG-based relevant verse selection with recent-verse avoidance
- LLM-backed responses (local Transformers or Ollama, with Grok as an option)
- TTS audio generation (pyttsx3) and verse image generation (Pillow)
- SQLite-backed users, history, analytics
- Ready-to-serve static media (audio/images)

Tech Stack
- FastAPI, Uvicorn
- SQLModel (SQLAlchemy) + SQLite by default
- transformers, sentence-transformers, torch
- Pillow, pyttsx3
- Optional: Ollama, Grok API

Prerequisites
- Python 3.10+ recommended
- Windows PowerShell (these instructions use PowerShell). On macOS/Linux, adapt activation commands accordingly
- Optional: GPU + CUDA for faster Transformers; Ollama installed if you want to use it

Quick Start (Windows PowerShell)
1) Clone and enter the project directory
   git clone <your-fork-or-repo-url>
   cd wisdomai

2) Create and activate a virtual environment
   python -m venv .venv
   . .venv\Scripts\Activate.ps1

3) Copy environment file (optional – defaults are safe to start)
   If a `.env` is missing but `env.example` exists, `start.py` will create one. You can also create `.env` manually (see Environment section below).

4) Start the server
   Option A (full startup helper – installs deps, inits DB, runs server):
     .venv\Scripts\python.exe start.py

   Option B (skip automatic installation and just run the app):
     python -m uvicorn main:app --host 0.0.0.0 --port 8000

   The API will be available at: http://localhost:8000
   Docs: http://localhost:8000/docs

Environment
The application loads settings from environment variables (via `.env`). Common variables and defaults:
- JWT_SECRET: change_this_secret_key_please
- DATABASE_URL: sqlite:///./god_ai.db
- MEDIA_ROOT: ./media
- USE_OLLAMA: false
- OLLAMA_URL: http://localhost:11434
- USE_GROK: false
- GROK_API_KEY: (empty)
- GROK_API_URL: https://api.x.ai/v1/chat/completions
- LLM_MODEL: distilgpt2

If you plan to use Ollama, set `USE_OLLAMA=true` and ensure Ollama is running locally. For Grok, set `USE_GROK=true` and provide `GROK_API_KEY`.

Embeddings and RAG
- Precomputed embeddings are included in `embeddings/` (verse_embeddings.pkl, verse_metadata.json, version.json)
- On startup, the RAG system loads these from disk. You can regenerate embeddings via the admin endpoint `/admin/regenerate-embeddings` (requires admin auth) or use scripts such as `generate_embeddings_final.py` if needed

Database
- Default: SQLite at `god_ai.db` in the project root
- Tables are auto-created on startup by `main.py`
- You can seed sample verses via the admin endpoint `/admin/seed-verses` (requires admin auth)
- `database_init.py` is used by `start.py` to initialize admin and seed data if you choose the helper path

Running With Docker (optional)
This repo includes `Dockerfile` and `docker-compose.yml`. Typical flow:
1) Build
   docker build -t god-ai-backend .

2) Run
   docker run -p 8000:8000 --env-file .env god-ai-backend

Or with Compose:
   docker compose up --build

API Overview (selected endpoints)
- POST /signup → returns JWT token
- POST /login → returns JWT token
- POST /chat → chat with mood detection and verse + LLM response (requires Bearer token)
- GET /daily-verse → daily personalized verse with media (requires token)
- POST /save-verse → save a verse to favorites (requires token)
- GET /profile → basic profile and recent history (requires token)
- GET /health → health check

Using the API (basic flow)
1) Sign up
   POST /signup
   body: { "name": "Your Name", "email": "you@example.com", "password": "secret" }

2) Use the returned `access_token` as a Bearer token for authorized endpoints (e.g., /chat, /daily-verse)

Media
- Generated audio files go to `media/audio`
- Generated images go to `media/images`
- The folder is served at `/media` (e.g., `/media/audio/<file>.mp3`)

LLM Backends
- Local Transformers: default if installed; uses `LLM_MODEL` (default `distilgpt2`)
- Ollama: set `USE_OLLAMA=true`, ensure Ollama is running and has a model (e.g., llama2 or mistral)
- Grok: set `USE_GROK=true` and provide `GROK_API_KEY`

Development Tips
- To speed up: you can skip `start.py` and run `uvicorn` directly while iterating
- Hot reload: use `--reload` with uvicorn in dev
   python -m uvicorn main:app --reload --port 8000

Testing
- A basic test script exists at `test_api.py` (adapt as needed)

Troubleshooting
- Port in use (Windows): If you see “only one usage of each socket address … is permitted,” stop the other server using port 8000 or change the port:
   python -m uvicorn main:app --port 8001

- Dependency install issues (Windows): Some ML packages (torch, bitsandbytes) may fail to install without proper build tools/CUDA. If you only want the app to run without ML acceleration, you can start with Option B (uvicorn) and rely on fallback/template responses.

- Ollama not reachable: Ensure Ollama is running and `OLLAMA_URL` is correct; otherwise the app falls back to local Transformers or template responses.

- TTS voice issues: `pyttsx3` uses system voices. If voice init fails, audio generation is skipped and endpoints still return data.

Project Scripts
- start.py: helper to create `.env`, install requirements, init DB, and run the server
- scripts/start_dev.py: convenience script for dev workflows

License
- Add your license here (e.g., MIT) if applicable

