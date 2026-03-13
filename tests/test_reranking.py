import pytest
from unittest.mock import MagicMock
from retrieval.reranking import RerankerService

def test_reranker_sigmoid_scores():
    """Verify that logit outputs from Cross-Encoder are correctly mapped to probabilities"""
    service = RerankerService()
    
    # Mock the internal sentence-transformers model
    mock_model = MagicMock()
    # Mock predicts logits: v1 logit = 2.0, v2 logit = -1.0
    mock_model.predict.return_value = [2.0, -1.0]
    
    service.model = mock_model
    service.enabled = True
    
    candidates = [
        {"verse_id": "v1", "text": "First passage", "similarity_score": 0.8},
        {"verse_id": "v2", "text": "Second passage", "similarity_score": 0.7}
    ]
    
    reranked = service.rerank("some query", candidates)
    
    assert len(reranked) == 2
    # Verify order: v1 has higher logit (2.0) -> higher probability -> sorted first
    assert reranked[0]["verse_id"] == "v1"
    assert reranked[1]["verse_id"] == "v2"
    
    # Sigmoid(2.0) is approx 0.8808
    assert reranked[0]["rerank_score"] == pytest.approx(0.8808, abs=1e-3)
    # Sigmoid(-1.0) is approx 0.2689
    assert reranked[1]["rerank_score"] == pytest.approx(0.2689, abs=1e-3)
