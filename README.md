# WisdomAI Backend – Setup and Usage Guide

WisdomAI is a production-grade spiritual AI companion built on FastAPI. It runs on a Retrieval-Augmented Generation (RAG) platform incorporating FAISS search indexes, Cross-Encoder reranking models, query rewriting, rolling conversation memory, structured citations, confidence scoring, hallucination fallbacks, RAGAS evaluations, Prometheus metrics, and Model Context Protocol (MCP) tool server interfaces.

---

## 🏛️ Architecture Overview

The codebase is refactored into a highly modular, decoupled architecture:
```
backend/
├── api/
│   ├── routes/              # Routing modules (auth, chat, verses, admin, evaluation)
│   ├── middleware/          # CORS policies and latency metrics middleware
│   └── dependencies/        # Auth & db injectors (get_current_user, get_session)
│
├── config/                  # Centralized Settings management (config/settings.py)
│
├── database/                # Connection logic (PostgreSQL + Neon configuration)
│
├── models/                  # SQLModel schemas (db_models.py)
│
├── schemas/                 # Pydantic validation schemas (api_schemas.py)
│
├── retrieval/               # FAISS vector engine, embeddings, and reranker
│
├── services/                # RAG, LLM, Memory, Safety, and MCP core layers
│
├── prometheus/              # Prometheus scrapers configuration
├── grafana/                 # Preconfigured Prometheus Grafana dashboards
├── tests/                   # Pytest unit and integration test suites
└── main.py                  # API entry point & lifecycle mountings
```

---

## ⚙️ Core Upgrades & Feature Enhancements

### 1. Vector Search Engine (FAISS & Cross-Encoder Reranker)
- **FAISS Engine**: Replaces pickle-based vector scans with an L2-normalized FAISS Index (`faiss.IndexFlatIP`) executing dot products to return exact cosine similarity.
- **Reranker**: Retrieved top 20 candidate passages are reranked using `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`, selecting the top 5 highest-quality passages.
- **Toggles**: Reranking can be dynamically bypassed through setting `ENABLE_RERANKING=False`.

### 2. Context-Aware Memory & Query Rewriting
- **Memory**: Stores raw message turns (`ChatMessage` table). Automatically runs context summarization utilizing the LLM if turns exceed a threshold, formatting queries as: `Summary + History + Context + User Query`.
- **Query Rewriting**: Analyzes user questions (e.g. "Why am I like this?") and transforms them into scriptural search keywords before searching FAISS.

### 3. Citations, Confidence Scoring, and Safety Guardrails
- **Citations**: Returns structured citations containing source names, verses, and custom locations (e.g. "Chapter 2 Verse 47", "John 3:16", "Surah 94:5") formatted in JSON.
- **Confidence**: Computes confidence scores (0-100) based on retrieval and reranking weights.
- **Safety checks**: If confidence is below 70, or safety checks fail, returns a graceful fallback message: *"I couldn't find reliable supporting passages for this question in our scriptural database."*

### 4. RAGAS Evaluations & Observability
- **Ragas**: Seeds a 100-question benchmark dataset. Measures faithfulness, answer relevancy, context recall, and context precision, logging runs under `EvaluationRun`.
- **Observability**: Latencies (embeddings, retrieval, rerank, LLM, total API) are tracked and exposed at `/metrics` for Prometheus scraping.

### 5. Model Context Protocol (MCP) Server
- Exposes standard stdio-based MCP tools (`search_scriptures`, `ask_contextual_question`, `retrieve_verse`) allowing external agents (like Claude Desktop) to connect.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or 3.11
- PostgreSQL database (or Neon account URL)
- Groq API Key (for LLM and evaluation judge)

### 2. Environment Variables (.env)
Create a `.env` file in the root directory:
```env
# Database Settings
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/wisdom_ai # Use Neon URL in production
USE_SQLITE_FALLBACK=false

# LLM & RAGAS Settings
USE_GROQ=true
GROQ_API_KEY=your-groq-api-key-here
GROQ_MODEL=llama-3.1-8b-instant

# RAG & Embeddings
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
ENABLE_RERANKING=true

# Prompt versioning (v1, v2, or v3)
PROMPT_VERSION=v1
```

### 3. Database Migration
WisdomAI uses SQLModel to automatically check tables and apply schemas on startup. To switch database providers (SQLite development fallback vs PostgreSQL/Neon production), toggle `DATABASE_URL` in your `.env`.

To seed the initial admin credentials and default verses:
```bash
python database_init.py
```

### 4. Local Development via Docker Compose
To boot up the complete stack (FastAPI Backend, PostgreSQL Database, Prometheus Scraper, and Grafana Dashboards):
```bash
docker compose up --build
```
- FastAPI Backend: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana Dashboards: `http://localhost:3001` (Default login: `admin` / `admin`)

---

## 🧪 Testing

A suite of unit and integration tests is provided under `tests/` using mocked model predictions for fast execution:
```bash
pytest tests/
```

---

## 🤖 MCP Server Setup

To connect Claude Desktop to WisdomAI as a tools provider, add this configuration to your Claude Desktop config file (usually `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):
```json
{
  "mcpServers": {
    "wisdom-ai": {
      "command": "python",
      "args": [
        "-m",
        "services.mcp.mcp_server"
      ],
      "env": {
        "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/wisdom_ai",
        "GROQ_API_KEY": "your-groq-api-key-here",
        "USE_GROQ": "true"
      }
    }
  }
}
```
