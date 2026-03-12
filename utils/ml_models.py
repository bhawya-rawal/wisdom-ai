import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Transformers pipeline loader
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class MLModelsLoader:
    """Manages lazy loading and thread-safe access to Hugging Face local pipelines"""
    
    def __init__(self):
        self._mood_pipeline = None
        self._summary_pipeline = None
        
        # Flag to disable heavy local models to speed up startup on lightweight instances
        self.enabled = TRANSFORMERS_AVAILABLE

    @property
    def mood_pipeline(self) -> Optional[Any]:
        """Lazy load the emotion detection pipeline"""
        if not self.enabled:
            return None
        if self._mood_pipeline is None:
            try:
                logger.info("Initializing emotion classification pipeline (j-hartmann/emotion-english-distilroberta-base)...")
                self._mood_pipeline = pipeline(
                    "text-classification", 
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None
                )
                logger.info("✓ Emotion classification pipeline loaded.")
            except Exception as e:
                logger.error("Failed to load emotion classification pipeline: %s", e)
                self._mood_pipeline = None
        return self._mood_pipeline

    @property
    def summary_pipeline(self) -> Optional[Any]:
        """Lazy load the text summarization pipeline"""
        if not self.enabled:
            return None
        if self._summary_pipeline is None:
            try:
                logger.info("Initializing text summarization pipeline (facebook/bart-large-cnn)...")
                self._summary_pipeline = pipeline(
                    "summarization", 
                    model="facebook/bart-large-cnn"
                )
                logger.info("✓ Text summarization pipeline loaded.")
            except Exception as e:
                logger.error("Failed to load text summarization pipeline: %s", e)
                self._summary_pipeline = None
        return self._summary_pipeline

# Global singleton loader
ml_loader = MLModelsLoader()
