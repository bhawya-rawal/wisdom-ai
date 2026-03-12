import logging
import time
from typing import Dict, Any

# Dynamic import to handle prometheus-client dependency gracefully
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Initialize metrics if Prometheus is available
if PROMETHEUS_AVAILABLE:
    # API queries count
    API_REQUESTS = Counter(
        "wisdom_ai_api_requests_total",
        "Total number of API requests handled",
        ["endpoint", "status"]
    )
    
    # Latency breakdown for each phase
    RAG_PIPELINE_LATENCY = Histogram(
        "wisdom_ai_pipeline_latency_seconds",
        "Latency breakdown of RAG stages",
        ["stage"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0, 30.0)
    )
    
    # Token usage tracker
    TOKEN_USAGE = Counter(
        "wisdom_ai_tokens_consumed_total",
        "Total number of LLM tokens consumed",
        ["model", "type"]  # type: input, output
    )
    
    # RAG confidence scores
    CONFIDENCE_SCORES = Histogram(
        "wisdom_ai_confidence_scores",
        "Distribution of RAG query confidence scores",
        buckets=(10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100)
    )
else:
    # Placeholders for non-prometheus setups
    API_REQUESTS = None
    RAG_PIPELINE_LATENCY = None
    TOKEN_USAGE = None
    CONFIDENCE_SCORES = None

class ObservabilityTracker:
    """Helper to track RAG metrics and API latency for logging and Prometheus metrics endpoint"""

    def track_request(self, endpoint: str, status_code: int):
        """Track incoming request status"""
        logger.info("Request to %s finished with status %d", endpoint, status_code)
        if PROMETHEUS_AVAILABLE and API_REQUESTS:
            API_REQUESTS.labels(endpoint=endpoint, status=str(status_code)).inc()

    def track_latency(self, stage: str, duration_seconds: float):
        """Record stage-specific duration in seconds"""
        logger.info("RAG Stage '%s' latency: %.4f seconds", stage, duration_seconds)
        if PROMETHEUS_AVAILABLE and RAG_PIPELINE_LATENCY:
            RAG_PIPELINE_LATENCY.labels(stage=stage).observe(duration_seconds)

    def track_tokens(self, model: str, input_tokens: int, output_tokens: int):
        """Record token consumption"""
        logger.info("Model %s consumed Input: %d | Output: %d tokens", model, input_tokens, output_tokens)
        if PROMETHEUS_AVAILABLE and TOKEN_USAGE:
            TOKEN_USAGE.labels(model=model, type="input").inc(input_tokens)
            TOKEN_USAGE.labels(model=model, type="output").inc(output_tokens)

    def track_confidence(self, score: float):
        """Record search result confidence distribution"""
        if PROMETHEUS_AVAILABLE and CONFIDENCE_SCORES:
            CONFIDENCE_SCORES.observe(score)

    def get_metrics_payload(self) -> tuple:
        """Expose Prometheus text format metrics for scraper endpoint"""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(), CONTENT_TYPE_LATEST
        return b"# Prometheus client library not installed.", "text/plain; version=0.0.4"

# Global metrics tracker
metrics_tracker = ObservabilityTracker()
