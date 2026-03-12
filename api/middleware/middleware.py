import time
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from services.monitoring.observability import metrics_tracker

def setup_cors(app):
    """Register CORS policy configuration on the FastAPI application instance"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for strict cloud production domains
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

async def track_latency_middleware(request: Request, call_next):
    """Custom middleware to measure total response latency of HTTP requests and export to observability"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    endpoint_path = request.url.path
    status_code = response.status_code
    
    # Track statistics
    metrics_tracker.track_request(endpoint_path, status_code)
    metrics_tracker.track_latency("total_api", duration)
    
    return response
