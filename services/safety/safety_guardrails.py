import logging
from typing import Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)

class SafetyGuardrails:
    """Validates generated spiritual responses for safety, hallucinations, and structural integrity"""

    def __init__(self):
        self.enabled = settings.ENABLE_GUARDRAILS
        self.fallback_response = settings.FALLBACK_RESPONSE
        
        # Simple local keywords to filter for basic safety guardrails
        self.unsafe_keywords = [
            "self-harm", "suicide", "kill myself", "bomb", "terrorist", 
            "hate speech", "illegal drugs", "hack", "exploit"
        ]

    def check_safety(self, text: str) -> bool:
        """Run keyword checks for safety violations"""
        text_lower = text.lower()
        for word in self.unsafe_keywords:
            if word in text_lower:
                logger.warning("Safety guardrail triggered by keyword: %s", word)
                return False
        return True

    def verify_citations(self, response: str, verse: Dict[str, Any]) -> bool:
        """Ensure the response contains the original text and reference of the source verse"""
        # Validate that a reasonable chunk of the verse text is in the response
        verse_text = verse.get("text", "")
        # Get first 30 characters of the verse to check presence
        verse_snippet = verse_text[:30].lower()
        
        if verse_snippet not in response.lower():
            logger.warning(
                "Citation guardrail triggered: Retrieved verse text snippet '%s' was not found in response.",
                verse_snippet
            )
            return False
            
        # Ensure the source book name (e.g., Gita, Quran, Bible) is referenced in the response
        source_name = verse.get("source", "").split("-")[0].strip().lower()
        if source_name not in response.lower():
            logger.warning("Citation guardrail triggered: Source '%s' is missing in response.", source_name)
            return False
            
        return True

    def validate(self, response: str, verse: Dict[str, Any], user_query: str) -> bool:
        """Combined validation checks. Returns True if response passes, False otherwise."""
        if not self.enabled:
            return True
            
        # 1. Safety Check
        if not self.check_safety(response) or not self.check_safety(user_query):
            return False
            
        # 2. Citation and Hallucination Check
        if not self.verify_citations(response, verse):
            return False
            
        return True

# Global safety guardrails instance
safety_guardrails = SafetyGuardrails()
