import time
import logging
import json
from typing import Dict, Any, List, Tuple, Optional

from config.settings import settings
from database.connection import get_session
from models.db_models import Verse
from retrieval.embeddings import embedding_service
from retrieval.faiss_engine import faiss_engine
from retrieval.reranking import reranker_service
from services.llm.llm_service import llm_service
from services.safety.safety_guardrails import safety_guardrails
from schemas.api_schemas import Citation

logger = logging.getLogger(__name__)

class RAGService:
    """Orchestrates the entire RAG pipeline from query rewriting to retrieval, reranking, safety checks, and response generation"""

    def _parse_citation_location(self, verse_id: str, source: str) -> str:
        """Helper to format locations like 'Chapter 2 Verse 47', 'John 3:16', 'Surah 94:5'"""
        if not verse_id:
            return "Unknown Location"
            
        parts = verse_id.split('_')
        
        # 1. Bhagavad Gita format: Gita_2.47 -> Chapter 2 Verse 47
        if "Gita" in verse_id or "gita" in verse_id.lower():
            if len(parts) > 1:
                chap_verse = parts[-1].split('.')
                if len(chap_verse) >= 2:
                    return f"Chapter {chap_verse[0]} Verse {chap_verse[1]}"
                return f"Verse {parts[-1]}"
            return "Bhagavad Gita"
            
        # 2. Bible format: Bible_KJV_19_19_2 -> Psalms 19:19 (or mapping values)
        elif "Bible" in verse_id:
            if len(parts) >= 4:
                # Bible_KJV_Book_Chapter_Verse
                chapter = parts[-2]
                verse = parts[-1]
                # Try to clean name e.g. "Bible - KJV" -> "John"
                book_name = source.replace("Bible - ", "").strip()
                return f"{book_name} {chapter}:{verse}"
            return "Bible"
            
        # 3. Quran format: Quran_94.5 -> Surah 94:5
        elif "Quran" in verse_id or "quran" in verse_id.lower():
            if len(parts) > 1:
                chap_verse = parts[-1].split('.')
                surah_num = chap_verse[0]
                ayah_num = chap_verse[1] if len(chap_verse) > 1 else ""
                
                surah_name = source.replace("Quran - ", "").strip()
                if surah_name and surah_name != "Quran":
                    return f"Surah {surah_name} {surah_num}:{ayah_num}"
                return f"Surah {surah_num}:{ayah_num}"
            return "Quran"
            
        return verse_id

    def calculate_confidence(self, sim_score: float, rerank_score: float) -> Tuple[float, str]:
        """Compute final confidence score (0-100) and return label"""
        if settings.ENABLE_RERANKING:
            # Weighted average: 30% FAISS Cosine Similarity, 70% Cross-Encoder relevance score
            conf = (0.3 * sim_score + 0.7 * rerank_score) * 100.0
        else:
            conf = sim_score * 100.0
            
        # Clip between 0 and 100
        conf = max(0.0, min(100.0, conf))
        
        if conf >= settings.CONFIDENCE_THRESHOLD_HIGH:
            label = "High"
        elif conf >= settings.CONFIDENCE_THRESHOLD_MEDIUM:
            label = "Medium"
        else:
            label = "Low"
            
        return round(conf, 1), label

    def answer_question(
        self, 
        user_query: str, 
        user_id: int, 
        mood: str, 
        history_summary: str,
        session
    ) -> Tuple[str, Dict[str, Any], Dict[str, float]]:
        """
        Executes query optimization, dense search, reranking, confidence scoring, 
        and response generation with safety verification.
        """
        latencies = {}
        
        # 1. Query Rewriting
        t0 = time.time()
        optimized_query = llm_service.rewrite_query(user_query, history_summary)
        latencies["rewrite_ms"] = round((time.time() - t0) * 1000.0, 1)
        
        # 2. Embedding generation
        t0 = time.time()
        query_embedding = embedding_service.get_embedding(optimized_query)
        latencies["embedding_ms"] = round((time.time() - t0) * 1000.0, 1)
        
        # 3. FAISS Retrieval (Top 20 candidates)
        t0 = time.time()
        candidates = faiss_engine.search(query_embedding, top_k=settings.RAG_TOP_K_FAISS, mood=mood)
        latencies["retrieval_ms"] = round((time.time() - t0) * 1000.0, 1)
        
        if not candidates:
            # No matching verses retrieved
            logger.warning("No candidate verses found during vector search.")
            return settings.FALLBACK_RESPONSE, {
                "verse_id": "None", "text": "", "source": "", 
                "citations": [], "confidence": 0.0, "confidence_label": "Low"
            }, latencies
            
        # 4. Cross-Encoder Reranking
        t0 = time.time()
        reranked_candidates = reranker_service.rerank(optimized_query, candidates)
        latencies["reranking_ms"] = round((time.time() - t0) * 1000.0, 1)
        
        # Select top passage
        top_verse = reranked_candidates[0]
        
        # 5. Compute Confidence Score
        confidence, confidence_label = self.calculate_confidence(
            top_verse.get("similarity_score", 0.5),
            top_verse.get("rerank_score", 0.5)
        )
        
        # Build Citations for Top 5 selected verses
        citations = []
        for v in reranked_candidates[:settings.RAG_TOP_K_FINAL]:
            loc = self._parse_citation_location(v["verse_id"], v["source"])
            citations.append(Citation(
                source=v["source"],
                location=loc,
                text=v["text"]
            ))
            
        # 6. Hallucination Threshold Check
        if confidence < settings.CONFIDENCE_THRESHOLD_MEDIUM:
            logger.warning("Confidence score (%.1f) is below threshold (70.0). Triggering safety fallback.", confidence)
            return settings.FALLBACK_RESPONSE, {
                "verse_id": top_verse["verse_id"],
                "text": top_verse["text"],
                "source": top_verse["source"],
                "citations": citations,
                "confidence": confidence,
                "confidence_label": confidence_label
            }, latencies

        # 7. Response Generation & Safety Retries
        t0 = time.time()
        reply = ""
        max_attempts = 2
        
        for attempt in range(1, max_attempts + 1):
            logger.info("Generating response (attempt %d/%d)...", attempt, max_attempts)
            reply = llm_service.generate_response(user_query, top_verse, mood, history_summary)
            
            # Guardrails validation
            if safety_guardrails.validate(reply, top_verse, user_query):
                logger.info("Response validation succeeded on attempt %d.", attempt)
                break
            else:
                logger.warning("Response validation failed on attempt %d.", attempt)
                if attempt == max_attempts:
                    logger.error("Guardrails validation failed after %d attempts. Returning fallback response.", max_attempts)
                    reply = settings.FALLBACK_RESPONSE
                    
        latencies["llm_generation_ms"] = round((time.time() - t0) * 1000.0, 1)
        
        verse_info = {
            "verse_id": top_verse["verse_id"],
            "text": top_verse["text"],
            "source": top_verse["source"],
            "citations": citations,
            "confidence": confidence,
            "confidence_label": confidence_label
        }
        
        return reply, verse_info, latencies

# Global RAG orchestration instance
rag_service = RAGService()
