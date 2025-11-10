"""
LLM Service Module
Provides integration with open-source LLMs for generating spiritual responses
"""

#generate response - by sending to either grok or ollama 
# prompt is written here 
# then the response by llm is made clean by fixing whitespaces
import os
import json
import re
import requests
from typing import Optional, Dict, Any
import time

# Local transformers support is intentionally not used anymore; Ollama/Groq
# are the only supported backends.
TRANSFORMERS_AVAILABLE = False

class LLMService:
    """Service for generating AI responses using open-source LLMs"""
    
    def __init__(self):
        self.model_name = os.getenv("LLM_MODEL", "distilgpt2")  # More suitable for text generation
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
        self.use_groq = os.getenv("USE_GROQ", "false").lower() == "true"
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_api_url = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
        self.local_model = None
        self.tokenizer = None
        
        # Initialize the appropriate model (Ollama first, then Groq).
        if self.use_ollama:
            self._init_ollama()
        if not self.use_ollama and self.use_groq and self.groq_api_key:
            self._init_groq()
        if not self.use_ollama and not self.use_groq:
            print("Warning: No LLM backend (Ollama/Groq) configured. LLMService will be unavailable.")
    
    def _init_groq(self):
        """Initialize Groq API connection"""
        try:
            if not self.groq_api_key:
                print("❌ Groq API key not provided")
                self.use_groq = False
                return
            
            print("✓ Groq API connection established")
            self.use_groq = True
            
        except Exception as e:
            print(f"❌ Groq initialization failed: {e}")
            self.use_groq = False
    
    def _init_ollama(self):
        """Initialize Ollama connection"""
        try:
            # Test if Ollama is running
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✓ Ollama connection established")
                self.use_ollama = True
            else:
                print("❌ Ollama not responding")
                self.use_ollama = False
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            self.use_ollama = False
    
    def generate_response(
        self, 
        user_message: str, 
        verse: Dict[str, Any], 
        mood: str, 
        user_summary: Optional[str] = None
    ) -> str:
        """
        Generate a personalized spiritual response using LLM
        
        Args:
            user_message: User's input message
            verse: Selected verse with text and source
            mood: Detected mood
            user_summary: Optional summary of user's message
            
        Returns:
            Generated response string
        """
        print(f"[DEBUG] use_groq={self.use_groq}, use_ollama={self.use_ollama}, local_model={self.local_model}")
        # Create context for the LLM
        context = self._create_context(user_message, verse, mood, user_summary)
        
        # Try Ollama first, then Groq. No template or local-model fallbacks.
        if self.use_ollama:
            return self._generate_with_ollama(context, verse)
        if self.use_groq:
            return self._generate_with_groq(context, verse)
        raise RuntimeError("No LLM backend (Ollama/Groq) is available.")
    
    def _create_context(self, user_message: str, verse: Dict[str, Any], mood: str, user_summary: Optional[str]) -> str:
        """Create context prompt for the LLM"""
        
        # Extract verse number from verse_id (e.g., "Gita_2.47" -> "2.47", "Bible_KJV_19_19_2" -> "19:19:2")
        verse_id = verse.get('verse_id', '')
        verse_number = self._extract_verse_number(verse_id, verse.get('source', ''))
        
        # Determine if we should acknowledge feelings
        has_emotions = mood not in ['neutral', 'joy', 'happy'] or any(word in user_message.lower() for word in ['sad', 'angry', 'fear', 'worried', 'anxious', 'depressed', 'hurt', 'pain', 'struggle', 'difficult'])
        
        # Build contextual intro
        if has_emotions:
            intro_instruction = f"Briefly acknowledge their {mood} feelings (1 sentence), then"
        else:
            intro_instruction = "Start directly with"
        
        prompt = f"""You are a warm, compassionate spiritual companion. The user said: "{user_message}"

You must share this verse: "{verse['text']}" — {verse['source']} {verse_number}

Write a warm response (200 words) that MUST include:
1. {intro_instruction} share the FULL verse text with verse number: "{verse['text']}" — {verse['source']} {verse_number}
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

        return prompt
    
    def _extract_verse_number(self, verse_id: str, source: str) -> str:
        """Extract verse number from verse_id for display"""
        if not verse_id:
            return ""
        
        # Handle different verse_id formats
        # Gita_2.47 -> 2.47
        if 'Gita' in verse_id or 'gita' in verse_id.lower():
            parts = verse_id.split('_')
            if len(parts) > 1:
                return parts[-1]  # e.g., "2.47"
        
        # Bible_KJV_19_19_2 -> 19:19:2 or Bible_KJV_19_19_2 -> 19:19
        elif 'Bible' in verse_id:
            parts = verse_id.split('_')
            if len(parts) >= 4:
                # Format: Book_Chapter_Verse or Book_Chapter_Verse_Subverse
                chapter = parts[-2] if len(parts) > 2 else ''
                verse = parts[-1] if len(parts) > 1 else ''
                if chapter and verse:
                    return f"{chapter}:{verse}"
        
        # Quran_7.69 -> 7:69
        elif 'Quran' in verse_id or 'quran' in verse_id.lower():
            parts = verse_id.split('_')
            if len(parts) > 1:
                verse_num = parts[-1].replace('.', ':')  # Convert 7.69 to 7:69
                return verse_num
        
        # Generic: try to extract numbers
        numbers = re.findall(r'\d+', verse_id)
        if numbers:
            if len(numbers) == 2:
                return f"{numbers[0]}:{numbers[1]}"
            elif len(numbers) == 1:
                return numbers[0]
        
        return ""
    
    def _format_paragraphs(self, text: str, verse: Optional[Dict[str, Any]] = None) -> str:
        """Format response with beautiful paragraph breaks"""
        if not text:
            return text
        
        # Ensure verse is on its own line
        if verse and verse.get('text'):
            verse_text = verse['text'][:50]  # First 50 chars for matching
            # If verse appears in text, ensure it's properly formatted
            verse_pattern = re.escape(verse_text)
            if re.search(verse_pattern, text, re.IGNORECASE):
                # Ensure verse has line breaks around it
                text = re.sub(
                    r'(".*?"\s*—\s*[^—\n]+(?:\s+\d+[:\d]*)?)',
                    r'\n\n\1\n\n',
                    text,
                    count=1
                )
        
        # Split into sentences and create natural paragraph breaks
        sentences = re.split(r'([.!?]\s+)', text)
        paragraphs = []
        current_para = []
        sentence_count = 0
        
        for i, part in enumerate(sentences):
            if part.strip():
                current_para.append(part)
                if part.strip().endswith(('.', '!', '?')):
                    sentence_count += 1
                    # Create paragraph break after 2-3 sentences
                    if sentence_count >= 2 and len(current_para) > 0:
                        para_text = ''.join(current_para).strip()
                        if para_text:
                            paragraphs.append(para_text)
                        current_para = []
                        sentence_count = 0
        
        # Add remaining sentences
        if current_para:
            para_text = ''.join(current_para).strip()
            if para_text:
                paragraphs.append(para_text)
        
        # Join paragraphs with double newlines
        formatted = '\n\n'.join(paragraphs)
        
        # Clean up any triple+ newlines
        formatted = re.sub(r'\n{3,}', '\n\n', formatted)
        
        return formatted.strip()
    
    def _clean_response(self, response: str, verse: Optional[Dict[str, Any]] = None) -> str:
        """Clean and trim the response to be concise and natural"""
        # Remove common verbose phrases
        verbose_phrases = [
            "Certainly, ",
            "Of course, ",
            "Yes, of course. ",
            "I am honored that ",
            "May I share with you ",
            "Here are some ",
            "Here's the verse:",
            "Here is a verse",
            "I hope this ",
            "Remember, ",
            "I understand how you're feeling. ",
            "I understand how you are feeling. ",
            "I understand. Here's a verse that might help:",
        ]
        
        cleaned = response.strip()
        
        # Remove verbose phrases
        for phrase in verbose_phrases:
            if cleaned.startswith(phrase):
                cleaned = cleaned[len(phrase):].strip()
        
        # Keep verse numbers but clean up formatting
        # Don't remove verse numbers - they should be preserved
        # Only remove parenthetical notes that aren't part of verse references
        cleaned = re.sub(r'\([^)]*\)(?!\s*[:\d])', '', cleaned)  # Remove parenthetical notes but keep verse refs
        
        # Ensure verse is included - if it's missing, don't trim too aggressively
        verse_text_short = verse.get('text', '')[:30] if verse and verse.get('text') else ''
        verse_markers = ['"', '—', verse_text_short] if verse_text_short else ['"', '—']
        has_verse = any(marker in cleaned for marker in verse_markers if marker)
        
        # Remove incomplete responses (ending with "Dear" or incomplete sentences)
        if cleaned.endswith("Dear") or cleaned.endswith("Dear [User]") or cleaned.endswith("—"):
            # Find last complete sentence
            sentences = cleaned.split('.')
            if len(sentences) > 1:
                # Keep all complete sentences except the last incomplete one
                cleaned = '. '.join(sentences[:-1]).strip()
                if cleaned and not cleaned.endswith('.'):
                    cleaned += '.'
        
        # Limit to reasonable length (about 200 words max, but ensure verse is included)
        words = cleaned.split()
        if len(words) > 200 and has_verse:
            # Find a good stopping point (end of sentence) but preserve the verse
            sentences = cleaned.split('.')
            trimmed = []
            word_count = 0
            verse_found = False
            for sentence in sentences:
                sentence_words = len(sentence.split())
                # If we've found the verse, allow more text after it
                if any(marker in sentence for marker in verse_markers if marker):
                    verse_found = True
                if word_count + sentence_words > (180 if verse_found else 150):
                    break
                trimmed.append(sentence)
                word_count += sentence_words
            cleaned = '. '.join(trimmed).strip()
            if cleaned and not cleaned.endswith('.'):
                cleaned += '.'
        elif len(words) > 200:
            # If no verse found, trim more aggressively
            sentences = cleaned.split('.')
            trimmed = sentences[:4]  # Keep first 4 sentences
            cleaned = '. '.join(trimmed).strip()
            if cleaned and not cleaned.endswith('.'):
                cleaned += '.'
        
        # Remove incomplete placeholders
        cleaned = re.sub(r'Dear \[?User\]?[,:]?', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'—\s*$', '', cleaned)  # Remove trailing source markers
        
        # Don't remove newlines here - let _format_paragraphs handle formatting
        # Just normalize excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Max 2 newlines
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Normalize spaces within lines
        
        # Ensure response has actual content, not just verse
        if cleaned.strip() and len(cleaned.split()) < 10:
            # Response is too short, might be incomplete
            return cleaned.strip() if cleaned.strip() else ""
        
        # Format with beautiful paragraphs
        cleaned = self._format_paragraphs(cleaned, verse)
        
        return cleaned.strip() if cleaned.strip() else ""
    
    def _generate_with_groq(self, context: str, verse: Dict[str, Any]) -> str:
        """Generate response using Groq API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            # Groq models: llama-3.1-8b-instant, mixtral-8x7b-32768, gemma-7b-it
            groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            
            payload = {
                "model": groq_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a compassionate spiritual AI companion. Provide warm, empathetic responses that explain verses and offer comfort."
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            response = requests.post(
                self.groq_api_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                cleaned = self._clean_response(content.strip(), verse)
                # Ensure verse is included with verse number
                verse_number = self._extract_verse_number(verse.get('verse_id', ''), verse.get('source', ''))
                verse_display = f'"{verse["text"]}"\n— {verse["source"]}'
                if verse_number:
                    verse_display += f' {verse_number}'
                
                if verse['text'] not in cleaned:
                    cleaned = f'{verse_display}\n\n{cleaned}'
                return cleaned
            else:
                print(f"Groq API error: {response.status_code} - {response.text}")
                raise RuntimeError(f"Groq API error: {response.status_code}")
                
        except Exception as e:
            print(f"Groq generation failed: {e}")
            raise
    
    def _generate_with_ollama(self, context: str, verse: Dict[str, Any]) -> str:
        """Generate response using Ollama API"""
        print("[DEBUG] Calling Ollama API with context:", context[:100], "...")
        try:
            # Use a smaller model that fits in available memory
            # tinyllama is ~637MB, much smaller than llama2:7b (3.8GB)
            model = os.getenv("OLLAMA_MODEL", "tinyllama")  # Default to tinyllama for low-memory systems
            payload = {
                "model": model,
                "prompt": context,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "num_predict": 250,  # Generate up to 250 tokens for a complete response
                    "stop": ["User:", "You:", "Dear [User]", "Dear User"],  # Stop at conversation breaks, not paragraph breaks
                    "repeat_penalty": 1.2  # Reduce repetition
                }
            }
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=90  # Increased timeout for testing
            )
            print(f"[DEBUG] Ollama API status: {response.status_code}")
            print(f"[DEBUG] Ollama API response: {response.text[:200]}")
            if response.status_code == 200:
                result = response.json()
                print("[DEBUG] Ollama API result:", result)
                raw_response = result.get("response", "I'm here to offer you comfort and guidance.")
                
                # Post-process to ensure concise, natural responses with verse included
                cleaned = self._clean_response(raw_response, verse)
                
                # Ensure verse is included - if not, add it
                verse_text_short = verse['text'][:30] if len(verse['text']) > 30 else verse['text']
                verse_number = self._extract_verse_number(verse.get('verse_id', ''), verse.get('source', ''))
                verse_display = f'"{verse["text"]}"\n— {verse["source"]}'
                if verse_number:
                    verse_display += f' {verse_number}'
                
                if verse['text'] not in cleaned and verse_text_short not in cleaned:
                    # Verse is missing, prepend it
                    cleaned = f'{verse_display}\n\n{cleaned}'
                
                # Check if response is incomplete (too short, ends with placeholder, or has no explanation)
                word_count = len(cleaned.split())
                is_incomplete = (word_count < 20 or 
                               cleaned.endswith('—') or 
                               'Dear' in cleaned[-30:] or
                               cleaned.count('.') < 2)  # Less than 2 complete sentences
                
                if is_incomplete:
                    # Response is incomplete, add a brief explanation
                    if verse['text'] in cleaned:
                        # Verse is there, just add explanation
                        brief_explanation = f"This verse speaks to moments of difficulty and reminds us that we are not alone in our struggles. It offers comfort and the promise that there is meaning even in our darkest times."
                        if not cleaned.rstrip().endswith('.'):
                            cleaned = cleaned.rstrip() + '.'
                        cleaned = f'{cleaned}\n\n{brief_explanation}'
                    else:
                        # Verse is missing, add both
                        brief_explanation = f"This verse speaks to moments of difficulty and reminds us that we are not alone in our struggles. It offers comfort and the promise that there is meaning even in our darkest times."
                        cleaned = f'{verse_display}\n\n{brief_explanation}'
                
                # Format with beautiful paragraphs
                cleaned = self._format_paragraphs(cleaned, verse)
                return cleaned
            else:
                print(f"[ERROR] Ollama API error: {response.status_code} - {response.text}")
                raise RuntimeError(f"Ollama API error: {response.status_code}")
        except Exception as e:
            print(f"[ERROR] Ollama generation failed: {e}")
            raise
    
    def _generate_with_transformers(self, context: str, verse: Dict[str, Any]) -> str:
        """Generate response using local transformers model"""
        raise RuntimeError("Local transformers backend is disabled. Use Ollama or Groq instead.")
    
    def _generate_template_response(self, user_message: str, verse: Dict[str, Any], mood: str) -> str:
        """Enhanced template-based response generation with verse explanation"""
        mood_responses = {
            "sadness": "I understand you're going through a difficult time. Let this verse bring you comfort and hope:",
            "happy": "I'm glad you're feeling joyful! Here's a verse to celebrate this moment:",
            "anger": "I sense you're feeling frustrated. This wisdom might help bring clarity and peace:",
            "fear": "I hear the worry in your words. Let this verse offer you strength and courage:",
            "surprise": "I can feel the surprise in your message. Here's some guidance for this moment:",
            "disgust": "I understand your concern. This verse might offer a different perspective:"
        }
        
        mood_intro = mood_responses.get(mood, "Here's a verse that might speak to your heart:")
        
        # Add verse explanation based on mood
        verse_explanations = {
            "sadness": "This verse reminds us that even in our darkest moments, there is comfort and strength available to us. It speaks to the healing power of faith and the promise that we are not alone in our struggles.",
            "happy": "This verse celebrates the joy and blessings in our lives. It reminds us to be grateful for the good moments and to share our happiness with others.",
            "anger": "This verse offers wisdom about managing difficult emotions. It reminds us that patience and understanding can help us navigate challenging situations with grace.",
            "fear": "This verse speaks to courage and trust. It reminds us that we have inner strength and that we can face our fears with faith and determination.",
            "surprise": "This verse offers guidance for unexpected moments. It reminds us that life's surprises can be opportunities for growth and learning.",
            "disgust": "This verse provides perspective on difficult situations. It reminds us that there are always different ways to view challenges and find meaning in them."
        }
        
        explanation = verse_explanations.get(mood, "This verse offers wisdom and guidance for your current situation. It reminds us that there is always hope and meaning to be found.")
        
        response = f"{mood_intro}\n\n\"{verse['text']}\"\n— {verse['source']}\n\n{explanation}\n\nHow does this verse speak to your heart today?"
        
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current LLM setup"""
        info = {
            "use_groq": self.use_groq,
            "use_ollama": self.use_ollama,
            "ollama_url": self.ollama_url if self.use_ollama else None,
            "local_model": self.model_name if self.local_model else None,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "backend": "groq" if self.use_groq else ("ollama" if self.use_ollama else "unavailable")
        }
        
        if self.use_ollama:
            try:
                response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    info["available_models"] = [model["name"] for model in models]
            except:
                info["available_models"] = []
        
        return info

# Global LLM service instance
llm_service = None

def initialize_llm():
    """Initialize the global LLM service"""
    global llm_service
    llm_service = LLMService()
    return llm_service

def get_llm_service():
    """Get the global LLM service instance"""
    return llm_service



