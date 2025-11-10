# Connecting Web Frontend to Backend

This guide explains how to connect the web folder (Next.js frontend) with the FastAPI backend.

## Quick Start

### Option 1: Development (Recommended)

1. **Start the Backend:**
   ```bash
   # From the root directory
   python start.py
   # Or manually:
   python -m uvicorn main:app --reload --port 8000
   ```
   The backend will be available at `http://localhost:8000`

2. **Start the Web Frontend:**
   ```bash
   # From the web directory
   cd web
   npm install  # First time only
   npm run dev
   ```
   The frontend will be available at `http://localhost:3000`

3. **Configure Environment:**
   - Copy `web/.env.example` to `web/.env.local`
   - The default `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000` should work for development
   - If your backend runs on a different port, update the URL accordingly

### Option 2: Docker Compose

1. **Start all services:**
   ```bash
   docker-compose up --build
   ```
   - Backend: `http://localhost:8000`
   - Frontend: `http://localhost:3000`
   - Database: `localhost:5432`

2. **Access the application:**
   - Open `http://localhost:3000` in your browser
   - The frontend will automatically connect to the backend

## Configuration

### Environment Variables

**Web Folder (`web/.env.local`):**
```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NODE_ENV=development
```

**Backend (`.env` in root):**
```env
JWT_SECRET=your_super_secret_jwt_key_change_this_in_production
DATABASE_URL=sqlite:///./god_ai.db
MEDIA_ROOT=./media
```

### How It Works

1. **API Routes:** The Next.js app has API routes in `web/app/api/` that proxy requests to the FastAPI backend
2. **Rewrites:** `next.config.js` automatically rewrites `/api/*` and `/media/*` to the backend
3. **CORS:** The backend is configured to accept requests from any origin (configure properly for production)

## API Endpoints

The web frontend connects to these backend endpoints:

- `POST /signup` - User registration
- `POST /login` - User login
- `POST /chat` - Chat with AI
- `GET /daily-verse` - Get daily verse
- `POST /save-verse` - Save a verse
- `GET /profile` - Get user profile
- `GET /my-saved-verses` - Get saved verses
- `GET /health` - Health check

## Troubleshooting

### Backend not reachable
- Check if backend is running: `curl http://localhost:8000/health`
- Verify `NEXT_PUBLIC_API_BASE_URL` in `web/.env.local`
- Check CORS settings in `main.py` (line 464-470)

### Frontend can't connect
- Ensure backend is running first
- Check browser console for errors
- Verify the API base URL is correct
- Check network tab in browser dev tools

### Port conflicts
- Backend default: 8000 (change in `main.py` or `start.py`)
- Frontend default: 3000 (change in `package.json` scripts)
- Update `NEXT_PUBLIC_API_BASE_URL` if backend port changes

## Production Setup

For production:

1. **Update CORS in `main.py`:**
   ```python
   allow_origins=["https://yourdomain.com", "https://www.yourdomain.com"]
   ```

2. **Set environment variables:**
   ```env
   NEXT_PUBLIC_API_BASE_URL=https://api.yourdomain.com
   ```

3. **Build and deploy:**
   ```bash
   cd web
   npm run build
   npm start
   ```

## Testing the Connection

1. Start the backend
2. Start the frontend
3. Visit `http://localhost:3000/api/health` - should return backend health status
4. Visit `http://localhost:3000` - should load the frontend

