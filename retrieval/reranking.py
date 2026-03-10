import logging
import math
from typing import List, Dict, Any
import numpy as np

# We import CrossEncoder dynamically to handle failure gracefully if the library is not set up
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

from config.settings import settings

logger = logging.getLogger(__name__)

class RerankerService:
    """Cross-Encoder reranking service to score search relevance of retrieved verses"""
    
    def __init__(self):
        self.enabled = settings.ENABLE_RERANKING
        self.model_name = settings.RERANKER_MODEL_NAME
        self.model = None
        
        if self.enabled and CROSS_ENCODER_AVAILABLE:
            try:
                self.model = CrossEncoder(self.model_name)
                logger.info("✓ CrossEncoder model loaded: %s", self.model_name)
            except Exception as e:
                logger.error("Failed to load CrossEncoder model %s: %s. Reranking will be disabled.", self.model_name, e)
                self.enabled = False
        else:
            if not CROSS_ENCODER_AVAILABLE:
                logger.warning("CrossEncoder library not available. Reranking disabled.")
            else:
                logger.info("Reranking is disabled by configuration.")

    def _sigmoid(self, x: float) -> float:
        """Map raw logit score to a [0, 1] probability range"""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank a list of candidates against the query using Cross-Encoder logits"""
        if not self.enabled or not self.model or not candidates:
            # Fallback or pass-through
            logger.info("Reranking bypassed or no candidates. Passing retrieved elements directly.")
            for cand in candidates:
                # If rerank is disabled, similarity score is used as final score
                cand["rerank_score"] = cand.get("similarity_score", 0.5)
            return candidates
            
        try:
            # Build input pairs for the Cross-Encoder model: [Query, Context Text]
            pairs = [[query, cand["text"]] for cand in candidates]
            
            # Predict logits
            logger.info("Running CrossEncoder inference on %d candidates...", len(pairs))
            logits = self.model.predict(pairs)
            
            # Log initial FAISS similarity scores and new CrossEncoder scores
            for i, cand in enumerate(candidates):
                logit = float(logits[i])
                prob = self._sigmoid(logit)
                
                cand["rerank_score"] = prob
                cand["raw_logit"] = logit
                
                logger.debug(
                    "Verse: %s | FAISS Similarity: %.4f | Rerank Score: %.4f (logit: %.4f)",
                    cand["verse_id"], cand["similarity_score"], prob, logit
                )
                
            # Sort candidates by rerank score descending
            reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
            
            logger.info(
                "Rerank complete. Top selection: %s (Score: %.4f, original similarity: %.4f)",
                reranked[0]["verse_id"], reranked[0]["rerank_score"], reranked[0]["similarity_score"]
            )
            return reranked
            
        except Exception as e:
            logger.error("Error during reranking calculation: %s. Falling back to FAISS retrieval sorting.", e)
            for cand in candidates:
                cand["rerank_score"] = cand.get("similarity_score", 0.5)
            return candidates

# Global reranker service instance
reranker_service = RerankerService()
