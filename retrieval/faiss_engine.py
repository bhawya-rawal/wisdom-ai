import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# We'll use a dynamic import for faiss to provide a robust numpy fallback if faiss isn't installed yet
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from config.settings import settings
from retrieval.embeddings import embedding_service

logger = logging.getLogger(__name__)

class FAISSEngine:
    """Vector search engine using FAISS with L2-normalized cosine similarity"""
    
    def __init__(self):
        self.index_dir = Path(settings.FAISS_INDEX_DIR)
        self.index_dir.mkdir(exist_ok=True)
        self.index_file = self.index_dir / "index.faiss"
        self.metadata_file = self.index_dir / "faiss_metadata.json"
        
        self.index = None
        self.index_to_verse_id: Dict[int, str] = {}
        self.verse_id_to_metadata: Dict[str, Dict[str, Any]] = {}
        
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        # Load or create index
        self.load_index()

    def _init_empty_index(self):
        """Initialize a new FAISS flat index using Inner Product (IP) for normalized cosine similarity"""
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("Initialized new FAISS IndexFlatIP of dimension %d", self.dimension)
        else:
            self.index = []  # List of dicts for numpy fallback
            logger.warning("FAISS not available. Initialized NumPy-based vector list.")

    def load_index(self) -> bool:
        """Load FAISS index and metadata from disk"""
        if not self.index_file.exists() or not self.metadata_file.exists():
            logger.warning("FAISS index or metadata files not found on disk. Creating empty index.")
            self._init_empty_index()
            return False
            
        try:
            # Load metadata
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                # JSON keys are strings; convert FAISS IDs back to integers
                self.index_to_verse_id = {int(k): v for k, v in meta.get("index_to_verse_id", {}).items()}
                self.verse_id_to_metadata = meta.get("verse_id_to_metadata", {})
                
            if FAISS_AVAILABLE:
                self.index = faiss.read_index(str(self.index_file))
                logger.info("Successfully loaded FAISS index from disk. Total verses: %d", self.index.ntotal)
            else:
                # Load embeddings as numpy fallback
                embeddings_pkl = self.index_dir / "fallback_embeddings.npy"
                if embeddings_pkl.exists():
                    self.index = np.load(str(embeddings_pkl)).tolist()
                else:
                    self.index = []
                logger.info("Loaded NumPy fallback index. Total verses: %d", len(self.index))
            return True
        except Exception as e:
            logger.error("Error loading FAISS index: %s. Reinitializing...", e)
            self._init_empty_index()
            return False

    def save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Save metadata
            meta = {
                "index_to_verse_id": {str(k): v for k, v in self.index_to_verse_id.items()},
                "verse_id_to_metadata": self.verse_id_to_metadata
            }
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)
                
            if FAISS_AVAILABLE and self.index is not None:
                faiss.write_index(self.index, str(self.index_file))
                logger.info("Saved FAISS index to disk. Total verses: %d", self.index.ntotal)
            else:
                # Save numpy fallback
                embeddings_pkl = self.index_dir / "fallback_embeddings.npy"
                if isinstance(self.index, list) and len(self.index) > 0:
                    np.save(str(embeddings_pkl), np.array(self.index))
                logger.info("Saved NumPy fallback index. Total verses: %d", len(self.index) if isinstance(self.index, list) else 0)
        except Exception as e:
            logger.error("Error saving FAISS index: %s", e)

    def add_verse(self, verse_id: str, text: str, source: str, mood_tags: List[str], embedding: np.ndarray):
        """Add a single verse embedding to FAISS"""
        # Ensure embedding is 1D or format correctly
        vector = np.array(embedding, dtype='float32').reshape(1, -1)
        
        # Add to index
        if FAISS_AVAILABLE:
            next_idx = self.index.ntotal
            self.index.add(vector)
        else:
            if not isinstance(self.index, list):
                self.index = []
            next_idx = len(self.index)
            self.index.append(vector[0])
            
        self.index_to_verse_id[next_idx] = verse_id
        self.verse_id_to_metadata[verse_id] = {
            "verse_id": verse_id,
            "text": text,
            "source": source,
            "mood_tags": mood_tags
        }

    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 20, 
        mood: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search top_k closest verses in FAISS index, with optional mood tag safety filtering"""
        total_indexed = self.index.ntotal if FAISS_AVAILABLE else len(self.index)
        if total_indexed == 0:
            logger.warning("Vector search called on an empty index.")
            return []
            
        top_k = min(top_k, total_indexed)
        q_vector = np.array(query_embedding, dtype='float32').reshape(1, -1)
        
        raw_results: List[Tuple[str, float]] = []
        
        if FAISS_AVAILABLE:
            # Query FAISS index (Inner Product on normalized vectors returns cosine similarity)
            distances, indices = self.index.search(q_vector, total_indexed)
            # FAISS returns arrays of shape (1, K)
            for dist, idx in zip(distances[0], indices[0]):
                if idx in self.index_to_verse_id:
                    v_id = self.index_to_verse_id[idx]
                    raw_results.append((v_id, float(dist)))
        else:
            # NumPy fallback logic
            vectors = np.array(self.index, dtype='float32')
            scores = np.dot(vectors, q_vector[0]) # Since vectors are already L2 normalized
            for idx, score in enumerate(scores):
                v_id = self.index_to_verse_id[idx]
                raw_results.append((v_id, float(score)))
                
        # Filter and sort
        filtered_results: List[Dict[str, Any]] = []
        for v_id, score in raw_results:
            metadata = self.verse_id_to_metadata.get(v_id)
            if not metadata:
                continue
                
            # Mood filtering (prevent conflicting emotions)
            if mood:
                v_moods = metadata.get("mood_tags", [])
                if v_moods:
                    mood_conflicts = {
                        'sadness': ['happy', 'joy', 'celebration'],
                        'happy': ['sadness', 'sorrow', 'grief'],
                        'angry': ['peace', 'calm', 'serenity'],
                        'fear': ['courage', 'strength', 'confidence']
                    }
                    conflicting = mood_conflicts.get(mood.lower(), [])
                    if any(tag in v_moods for tag in conflicting):
                        continue
            
            # Map score to [0, 1] range (FAISS IP of normalized vectors ranges from -1 to 1)
            cosine_score = (score + 1.0) / 2.0 if score < 1.0 else 1.0
            
            filtered_results.append({
                "verse_id": v_id,
                "text": metadata["text"],
                "source": metadata["source"],
                "mood_tags": metadata["mood_tags"],
                "similarity_score": cosine_score
            })
            
        # Return top_k matching candidates
        # Sorted by similarity score descending
        filtered_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return filtered_results[:top_k]

    def rebuild_index(self, db_session) -> int:
        """Fetch all verses from database, compute normalized embeddings, rebuild and save the FAISS index"""
        logger.info("Starting FAISS index rebuild...")
        from models.db_models import Verse
        from sqlmodel import select
        
        # Reset current index representation
        self._init_empty_index()
        self.index_to_verse_id = {}
        self.verse_id_to_metadata = {}
        
        verses = db_session.exec(select(Verse)).all()
        if not verses:
            logger.warning("No verses found in DB to index.")
            return 0
            
        # Prepare batch vectors to run inference quickly
        texts_to_embed = []
        verse_data_list = []
        
        for verse in verses:
            try:
                mood_tags = json.loads(verse.mood_tags or "[]")
            except Exception:
                mood_tags = []
                
            # Embed format: Text + Source + Mood tags
            searchable_text = f"{verse.text} {verse.source} {' '.join(mood_tags)}"
            texts_to_embed.append(searchable_text)
            verse_data_list.append((verse.verse_id, verse.text, verse.source, mood_tags))
            
        # Batch embedding calculation
        logger.info("Computing embeddings for %d verses...", len(texts_to_embed))
        embeddings = embedding_service.get_embeddings(texts_to_embed)
        
        for idx, (v_id, text, source, mood_tags) in enumerate(verse_data_list):
            emb = embeddings[idx]
            self.add_verse(v_id, text, source, mood_tags, emb)
            
        self.save_index()
        total_added = len(self.index_to_verse_id)
        logger.info("FAISS index rebuild finished successfully. Total indexed: %d", total_added)
        return total_added

# Global vector engine instance
faiss_engine = FAISSEngine()
