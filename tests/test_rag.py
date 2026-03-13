import pytest
from unittest.mock import MagicMock, patch
from services.rag.rag_service import rag_service
from config.settings import settings

def test_confidence_calculations():
    """Verify confidence score calculations and thresholds"""
    # 1. High confidence case
    score, label = rag_service.calculate_confidence(0.9, 0.9)
    assert score == 90.0
    assert label == "High"
    
    # 2. Medium confidence case
    score, label = rag_service.calculate_confidence(0.75, 0.75)
    assert score == 75.0
    assert label == "Medium"
    
    # 3. Low confidence case
    score, label = rag_service.calculate_confidence(0.6, 0.6)
    assert score == 60.0
    assert label == "Low"

@patch("services.rag.rag_service.embedding_service")
@patch("services.rag.rag_service.faiss_engine")
@patch("services.rag.rag_service.reranker_service")
@patch("services.rag.rag_service.llm_service")
def test_low_confidence_hallucination_fallback(mock_llm, mock_rerank, mock_faiss, mock_embed):
    """Verify that low-confidence searches return fallback instead of hallucinating answers"""
    mock_embed.get_embedding.return_value = [0.1] * 384
    
    # Simulate a low similarity score search result
    mock_faiss.search.return_value = [
        {"verse_id": "v1", "text": "God is our strength", "source": "Bible", "similarity_score": 0.4}
    ]
    mock_rerank.rerank.return_value = [
        {"verse_id": "v1", "text": "God is our strength", "source": "Bible", "similarity_score": 0.4, "rerank_score": 0.3}
    ]
    
    mock_session = MagicMock()
    
    reply, verse_info, _ = rag_service.answer_question(
        user_query="How do I make a million dollars?",
        user_id=1,
        mood="neutral",
        history_summary="",
        session=mock_session
    )
    
    # Assert fallback triggers
    assert reply == settings.FALLBACK_RESPONSE
    # Assert LLM was never invoked because search confidence was too low
    mock_llm.generate_response.assert_not_called()
