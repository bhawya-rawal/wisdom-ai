import pytest
import numpy as np
from unittest.mock import MagicMock
from retrieval.faiss_engine import FAISSEngine

def test_faiss_engine_normalization():
    """Verify that embeddings are correctly added and searched with cosine similarity"""
    engine = FAISSEngine()
    
    # Reset index manually
    engine._init_empty_index()
    engine.index_to_verse_id = {}
    engine.verse_id_to_metadata = {}
    
    # Add dummy vectors
    vec1 = np.array([1.0, 0.0] + [0.0] * 382, dtype='float32') # Normalized
    vec2 = np.array([0.0, 1.0] + [0.0] * 382, dtype='float32') # Normalized
    
    engine.add_verse("v1", "Love is patient", "Bible", ["love"], vec1)
    engine.add_verse("v2", "Seek knowledge", "Quran", ["wisdom"], vec2)
    
    # Search with query vector pointing directly at vec1
    query_vec = np.array([1.0, 0.0] + [0.0] * 382, dtype='float32')
    results = engine.search(query_vec, top_k=2)
    
    assert len(results) == 2
    assert results[0]["verse_id"] == "v1"
    # Cosine score range mapping is (score+1)/2. Vec1 matches query perfectly, so score is 1.0. Range maps to 1.0.
    assert results[0]["similarity_score"] == pytest.approx(1.0)
    
    # Vec2 is orthogonal, dot product = 0. Cosine score is 0. Range maps (0+1)/2 = 0.5.
    assert results[1]["verse_id"] == "v2"
    assert results[1]["similarity_score"] == pytest.approx(0.5)
