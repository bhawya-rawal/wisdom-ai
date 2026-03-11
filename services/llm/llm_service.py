import os
import re
import logging
import requests
from typing import Optional, Dict, Any, List
from config.settings import settings

logger = logging.getLogger(__name__)

# Prompt version definitions
PROMPTS = {
    "v1": {
        "system": "You are a warm, compassionate spiritual companion. Provide empathetic comfort using scripture.",
        "user_template": """You are a warm, compassionate spiritual companion. The user said: "{user_message}"

You must share this verse: "{verse_text}" — {verse_source} {verse_number}

Write a warm response (200 words) that MUST include:
1. {intro_instruction} share the FULL verse text with verse number: "{verse_text}" — {verse_source} {verse_number}
2. A brief, simple explanation of what the verse means and how it relates to their message (2-3 sentences)
3. A short encouraging closing (1 sentence)

CRITICAL RULES:
- ALWAYS include the complete verse text WITH verse number
- ALWAYS format: "verse text" — Source verse_number (e.g., "text" — Bible - KJV 7:14)
- Format your response in MULTIPLE PARAGRAPHS (2-3 paragraphs)
- Put the verse in its own paragraph with proper line breaks
- Use paragraph breaks to make the response visually appealing
- Keep it warm and conversational, not formal
- NO phrases like "I understand how you're feeling" or "Certainly" or "Of course"
- NO repetitive acknowledgments
- Brief explanation only - don't over-explain
"""
    },
    "v2": {
        "system": "You are a modern, practical spiritual guide. Offer actionable, scripture-supported advice.",
        "user_template": """The user is seeking practical guidance: "{user_message}"

Share this wisdom verse: "{verse_text}" — {verse_source} {verse_number}

Write a structured, modern response (200 words):
1. Acknowledge their situation in a warm, direct way.
2. Present the verse on its own line: "{verse_text}" — {verse_source} {verse_number}
3. Explain 2 practical takeaways or steps they can take today inspired by this verse.
4. Close with a short word of encouragement.

Keep it structured, clear, and actionable. Avoid flowery language or generic padding.
"""
    },
    "v3": {
        "system": "You are a deeply philosophical spiritual mentor. Inspire introspection and contemplative meditation.",
        "user_template": """The user is reflecting: "{user_message}"

Share this deep verse: "{verse_text}" — {verse_source} {verse_number}

Write a contemplative response (200 words):
1. Reflect on the philosophical and emotional depth of their current state.
2. Present the verse: "{verse_text}" — {verse_source} {verse_number}
3. Pose one deep reflective question for them to meditate on, linking the scriptural context with their reflection.
4. Close with a peaceful benediction.

Keep the tone profound, quiet, meditative, and deep.
"""
    }
}

class LLMService:
    """Service for interacting with Groq Cloud or Ollama local inference engines"""
    
    def __init__(self):
        self.use_groq = settings.USE_GROQ
        self.groq_api_key = settings.GROQ_API_KEY
        self.groq_url = settings.GROQ_API_URL
        self.groq_model = settings.GROQ_MODEL
        
        self.use_ollama = settings.USE_OLLAMA
        self.ollama_url = settings.OLLAMA_URL
        self.ollama_model = settings.OLLAMA_MODEL
        
        self.prompt_version = settings.PROMPT_VERSION
        if self.prompt_version not in PROMPTS:
            logger.warning("Unsupported prompt version '%s'. Falling back to 'v1'.", self.prompt_version)
            self.prompt_version = "v1"

    def _call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
        """Central client multiplexer between Groq API and local Ollama API"""
        if self.use_groq and self.groq_api_key:
            return self._call_groq(system_prompt, user_prompt, max_tokens)
        elif self.use_ollama:
            return self._call_ollama(system_prompt, user_prompt, max_tokens)
        else:
            raise RuntimeError("No LLM backend (Groq/Ollama) configured or API key missing.")

    def _call_groq(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.groq_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }
        try:
            response = requests.post(self.groq_url, json=payload, headers=headers, timeout=20)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                logger.error("Groq API error: %d - %s", response.status_code, response.text)
                raise RuntimeError(f"Groq API Error: {response.status_code}")
        except Exception as e:
            logger.error("Failed to connect to Groq: %s", e)
            raise e

    def _call_ollama(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        # Combine prompts for Ollama single-prompt completion endpoints
        full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        payload = {
            "model": self.ollama_model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": max_tokens
            }
        }
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=90)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                logger.error("Ollama API error: %d - %s", response.status_code, response.text)
                raise RuntimeError(f"Ollama API Error: {response.status_code}")
        except Exception as e:
            logger.error("Failed to connect to Ollama: %s", e)
            raise e

    def rewrite_query(self, query: str, history_summary: str = "") -> str:
        """Optimize user query into scriptural concepts and emotional themes for FAISS vector retrieval"""
        system_msg = "You are a precise search query optimizer. Given a user's statement and conversation history, output a list of related spiritual topics, emotions, and concepts (nouns/keywords). DO NOT explain, DO NOT write full sentences, DO NOT output introductory filler. ONLY output a comma-separated list of keywords."
        user_msg = f"Optimize this query for scriptural database lookup: \"{query}\""
        if history_summary:
            user_msg += f"\nConversation Context Summary: {history_summary}"
            
        try:
            optimized = self._call_llm(system_msg, user_msg, max_tokens=50)
            # Remove any wrapping quotes or prefix statements
            clean = re.sub(r'^(Optimized query:|Keywords:|\s*-\s*|\d+\.\s*)', '', optimized, flags=re.IGNORECASE).strip()
            # If the LLM returned empty, fallback
            if clean and len(clean) > 2:
                logger.info("Rewrote query: \"%s\" -> \"%s\"", query, clean)
                return f"{query} {clean}"
            return query
        except Exception as e:
            logger.warning("Query rewriting failed: %s. Falling back to original query.", e)
            return query

    def summarize_history(self, summary: str, new_messages: List[Dict[str, str]]) -> str:
        """Construct or update a conversation summary utilizing the LLM"""
        system_msg = "You are a summarizing assistant. Write a concise, single-paragraph summary of the conversation context so far. Focus on the user's feelings, questions, and progress."
        
        chat_str = ""
        for msg in new_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            chat_str += f"{role.upper()}: {content}\n"
            
        user_msg = f"Previous Summary: {summary}\n\nNew messages:\n{chat_str}\nProvide an updated, single-paragraph summary."
        try:
            new_summary = self._call_llm(system_msg, user_msg, max_tokens=150)
            logger.info("Created new conversation summary: %s", new_summary[:100])
            return new_summary.strip()
        except Exception as e:
            logger.warning("Conversation summarization failed: %s. Preserving current summary.", e)
            return summary

    def _extract_verse_number(self, verse_id: str) -> str:
        """Extract verse number from verse_id for display"""
        if not verse_id:
            return ""
        if 'Gita' in verse_id or 'gita' in verse_id.lower():
            parts = verse_id.split('_')
            return parts[-1] if len(parts) > 1 else ""
        elif 'Bible' in verse_id:
            parts = verse_id.split('_')
            if len(parts) >= 4:
                chapter = parts[-2]
                verse = parts[-1]
                return f"{chapter}:{verse}"
        elif 'Quran' in verse_id or 'quran' in verse_id.lower():
            parts = verse_id.split('_')
            if len(parts) > 1:
                return parts[-1].replace('.', ':')
        
        numbers = re.findall(r'\d+', verse_id)
        if numbers:
            return f"{numbers[0]}:{numbers[1]}" if len(numbers) >= 2 else numbers[0]
        return ""

    def _clean_response(self, response: str, verse_text: str, verse_source: str, verse_number: str) -> str:
        """Post-process and clean LLM responses to ensure structured format and clear paragraphs"""
        cleaned = response.strip()
        
        # Remove verbose boilerplate beginnings
        boilerplate = [
            "Certainly, ", "Of course, ", "Yes, of course. ", "I understand how you're feeling. ",
            "I understand. Here's a verse that might help:", "Here is a verse that aligns with your feelings:"
        ]
        for phrase in boilerplate:
            if cleaned.startswith(phrase):
                cleaned = cleaned[len(phrase):].strip()
                
        # Format code clean paragraph breaks
        # Split into sentences and create natural paragraph breaks
        sentences = re.split(r'([.!?]\s+)', cleaned)
        paragraphs = []
        current_para = []
        sentence_count = 0
        
        # Ensure verse is standalone
        verse_marker = f'"{verse_text}"\n— {verse_source} {verse_number}'
        
        for part in sentences:
            if part.strip():
                current_para.append(part)
                if part.strip().endswith(('.', '!', '?')):
                    sentence_count += 1
                    if sentence_count >= 2:
                        para_text = ''.join(current_para).strip()
                        if para_text:
                            paragraphs.append(para_text)
                        current_para = []
                        sentence_count = 0
                        
        if current_para:
            para_text = ''.join(current_para).strip()
            if para_text:
                paragraphs.append(para_text)
                
        # Combine
        formatted = '\n\n'.join(paragraphs)
        # Clean triple newlines
        formatted = re.sub(r'\n{3,}', '\n\n', formatted)
        
        return formatted.strip()

    def generate_response(
        self, 
        user_message: str, 
        verse: Dict[str, Any], 
        mood: str, 
        history_summary: Optional[str] = None
    ) -> str:
        """Compose a structured scriptural guidance response utilizing the configured prompt version"""
        p_info = PROMPTS[self.prompt_version]
        system_prompt = p_info["system"]
        template = p_info["user_template"]
        
        verse_number = self._extract_verse_number(verse.get("verse_id", ""))
        
        # v1 specific instructions
        has_emotions = mood not in ['neutral', 'joy', 'happy'] or any(
            w in user_message.lower() for w in ['sad', 'angry', 'fear', 'worried', 'anxious', 'depressed', 'hurt', 'pain']
        )
        intro_instruction = f"Briefly acknowledge their {mood} feelings (1 sentence), then" if has_emotions else "Start directly with"
        
        # Format the user instruction
        user_prompt = template.format(
            user_message=user_message,
            verse_text=verse["text"],
            verse_source=verse["source"],
            verse_number=verse_number,
            intro_instruction=intro_instruction
        )
        
        if history_summary:
            user_prompt = f"Conversation Summary Context: {history_summary}\n\n{user_prompt}"
            
        logger.info("Generating response using prompt version: %s", self.prompt_version)
        response_raw = self._call_llm(system_prompt, user_prompt)
        
        # Clean response
        cleaned = self._clean_response(response_raw, verse["text"], verse["source"], verse_number)
        
        # Verify the verse is in the output; if not, prepend it
        short_verse = verse["text"][:30]
        if short_verse not in cleaned:
            verse_display = f'"{verse["text"]}"\n— {verse["source"]} {verse_number}'
            cleaned = f'{verse_display}\n\n{cleaned}'
            
        return cleaned

# Global instance of LLMService
llm_service = LLMService()
