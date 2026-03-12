import os
import logging
from PIL import Image, ImageDraw, ImageFont
from config.settings import settings

logger = logging.getLogger(__name__)

# Dynamic imports of TTS engines to handle environment variations cleanly
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

def generate_tts(verse_text: str, verse_id: str) -> str:
    """Generate TTS audio file for the verse text, using Hindi or English-Indian voices"""
    try:
        audio_filename = f"{verse_id.replace(' ', '_').replace('.', '_')}.mp3"
        audio_dir = os.path.join(settings.MEDIA_ROOT, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        audio_path = os.path.join(audio_dir, audio_filename)
        
        # 1. Try gTTS
        if GTTS_AVAILABLE:
            try:
                # Detect Hindi text characters to apply correct Hindi voice models
                has_hindi = any('\u0900' <= char <= '\u097F' for char in verse_text)
                if has_hindi:
                    tts = gTTS(text=verse_text, lang='hi', slow=True)
                else:
                    tts = gTTS(text=verse_text, lang='en', slow=True, tld='co.in')
                tts.save(audio_path)
                logger.info("Generated TTS audio via gTTS: %s", audio_path)
                return f"/media/audio/{audio_filename}"
            except Exception as e:
                logger.warning("gTTS generation failed, attempting pyttsx3 fallback: %s", e)

        # 2. Try pyttsx3 Fallback
        if pyttsx3:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if 'male' in voice.name.lower() or 'david' in voice.name.lower() or 'zira' not in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
                else:
                    engine.setProperty('voice', voices[0].id)
            
            engine.setProperty('rate', 120)  # Slower contemplative rate
            engine.setProperty('volume', 0.95)
            
            try:
                engine.setProperty('pitch', 0.7)  # Lower pitch for deeper tone
            except Exception:
                pass
                
            engine.save_to_file(verse_text, audio_path)
            engine.runAndWait()
            logger.info("Generated TTS audio via pyttsx3 fallback: %s", audio_path)
            return f"/media/audio/{audio_filename}"
            
        logger.error("No TTS engine available (neither gTTS nor pyttsx3 could run)")
        return ""
    except Exception as e:
        logger.error("TTS generation wrapper failed completely: %s", e)
        return ""

def generate_verse_image(verse_text: str, verse_id: str) -> str:
    """Create an attractive social image representing the spiritual passage"""
    try:
        width, height = 1200, 630
        img = Image.new("RGB", (width, height), color=(250, 245, 240))
        draw = ImageDraw.Draw(img)
        
        # Background gradient
        for y in range(height):
            color_val = int(250 - (y / height) * 30)
            draw.line([(0, y), (width, y)], fill=(color_val, color_val - 5, color_val - 10))
            
        # Load fonts
        try:
            font_large = ImageFont.truetype("arial.ttf", 32)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
            
        margin = 80
        max_width = width - (margin * 2)
        
        words = verse_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if font_large:
                bbox = draw.textbbox((0, 0), test_line, font=font_large)
                text_width = bbox[2] - bbox[0]
            else:
                text_width = len(test_line) * 10
                
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    lines.append(word)
        if current_line:
            lines.append(current_line)
            
        line_height = 45
        start_y = height // 2 - (len(lines) * line_height) // 2
        
        for i, line in enumerate(lines):
            y_pos = start_y + (i * line_height)
            if font_large:
                bbox = draw.textbbox((0, 0), line, font=font_large)
                text_width = bbox[2] - bbox[0]
            else:
                text_width = len(line) * 10
                
            x_pos = (width - text_width) // 2
            draw.text((x_pos, y_pos), line, font=font_large, fill=(40, 40, 40))
            
        draw.rectangle([(margin - 10, margin - 10), (width - margin + 10, height - margin + 10)], 
                       outline=(200, 180, 160), width=3)
                       
        # Footer branding
        footer_text = "WisdomAI 🌸"
        if font_small:
            bbox = draw.textbbox((0, 0), footer_text, font=font_small)
            footer_width = bbox[2] - bbox[0]
        else:
            footer_width = len(footer_text) * 8
            
        draw.text((width - footer_width - margin, height - 40), footer_text, font=font_small, fill=(120, 120, 120))
        
        images_dir = os.path.join(settings.MEDIA_ROOT, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        image_filename = f"{verse_id.replace(' ', '_').replace('.', '_')}.png"
        image_path = os.path.join(images_dir, image_filename)
        img.save(image_path, "PNG")
        
        return f"/media/images/{image_filename}"
    except Exception as e:
        logger.error("Image card generation failed: %s", e)
        return ""
