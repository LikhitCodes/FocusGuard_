"""
FocusGuard: Privacy-Focused AI Distraction Blocker
-------------------------------------------------
An AI-powered application that monitors screen activity, classifies productivity,
and helps maintain focus through smart interventions while respecting privacy.
"""

import os
import time
import json
import logging
import platform
import subprocess
import re
import functools
import threading
import queue
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import openai

import pyautogui
import pytesseract
from PIL import Image
import numpy as np
import cv2
import requests
import tempfile
import textwrap
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.utils import which
load_dotenv()


AudioSegment.converter = r"D:\Softwares\ffmpeg-2025-09-28-git-0fdb5829e3-full_build\ffmpeg-2025-09-28-git-0fdb5829e3-full_build\bin"

# Optional audio dependencies - gracefully handle if not available
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    logging.warning("gtts not available - voice alerts will be disabled")

try:
    from pydub import AudioSegment
    from pydub.playback import play
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logging.warning("pydub not available - audio playback will be disabled")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("focus_guard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FocusGuard")

print("AKASH_API_KEY from .env:", os.getenv("AKASH_API_KEY"))

# Configuration
DEFAULT_CONFIG = {
    "screenshot_interval": 8,  # seconds
    "ocr_psm_mode": 6,
    "max_text_length": 500,  # token limit for API
    "distraction_threshold": 3,  # consecutive distractions before intervention
    "api_timeout": 10,  # seconds
    "notification_frequency": 2,  # minutes
    "enable_voice_alerts": True,  # Will auto-disable if dependencies missing
    "enable_app_blocking": True,
    "privacy": {
        "store_screenshots": False,
        "store_raw_text": False,
        "anonymize_personal_data": True
    },
    "distractions": {
        "social_media": ["instagram.com", "facebook.com", "twitter.com", "tiktok.com", "reddit.com"],
        "entertainment": ["netflix.com", "youtube.com/watch", "youtube.com/", "twitch.tv", "hulu.com", "disneyplus.com", "hbomax.com", "primevideo.com"],
        "games": ["steam", "epic games", "battle.net"]
    },
    "productivity": {
        "work": ["vscode", "terminal", "cmd.exe", "powershell", "excel", "word", "powerpoint", "google docs","overleaf","overleaf.com"],
        "study": ["pdf", "research", "lecture", "coursera", "edx", "khan academy"],
        "communication": ["outlook", "gmail", "calendar", "meets", "zoom"]
    }
}

# ===== SCREEN MONITORING MODULE =====

class ScreenMonitor:
    """Handles screenshot capture and text extraction."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.last_capture_time = 0
        self.screenshot_interval = config.get("screenshot_interval", 8)
        self.ocr_psm_mode = config.get("ocr_psm_mode", 6)
        
        # Ensure tesseract is properly configured
        if platform.system() == "Windows":
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    def capture_screen(self) -> Image.Image:
        """Capture the current screen."""
        screenshot = pyautogui.screenshot()
        return screenshot
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to grayscale
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get black and white image
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        return thresh
    
    def extract_text(self, image: np.ndarray) -> str:
        """Extract text from the preprocessed image."""
        custom_config = f'--psm {self.ocr_psm_mode}'
        text = pytesseract.image_to_string(image, config=custom_config)
        return text
    
    def get_active_window_text(self) -> str:
        """Get text from active window with rate limiting."""
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_capture_time < self.screenshot_interval:
            time_to_wait = self.screenshot_interval - (current_time - self.last_capture_time)
            time.sleep(max(0, time_to_wait))
        
        # Capture and process
        screenshot = self.capture_screen()
        preprocessed = self.preprocess_image(screenshot)
        text = self.extract_text(preprocessed)
        
        # Clean and limit text
        cleaned_text = self.clean_text(text)
        
        # Update timestamp
        self.last_capture_time = time.time()
        
        return cleaned_text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Filter out common non-informative elements
        text = re.sub(r'[^\w\s.,?!:\-]', '', text)
        
        # Limit text length
        if len(text) > self.config.get("max_text_length", 500):
            text = text[:self.config.get("max_text_length", 500)]
        
        return text

    def get_active_window_info(self) -> Dict[str, str]:
        """Get both text content and window title"""
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_capture_time < self.screenshot_interval:
            time_to_wait = self.screenshot_interval - (current_time - self.last_capture_time)
            time.sleep(max(0, time_to_wait))
        
        # Get window title (platform specific)
        window_title = ""
        try:
            if platform.system() == "Windows":
                import win32gui
                window_title = win32gui.GetWindowText(win32gui.GetForegroundWindow())
            elif platform.system() == "Darwin":
                window_title = subprocess.run(
                    ["osascript", "-e", 'tell app "System Events" to get name of first process whose frontmost is true'],
                    capture_output=True, text=True
                ).stdout.strip()
            else:  # Linux
                window_title = subprocess.run(
                    ["xdotool", "getwindowfocus", "getwindowname"],
                    capture_output=True, text=True
                ).stdout.strip()
        except Exception as e:
            logger.warning(f"Could not get window title: {e}")
        
        # Get screen content
        screenshot = self.capture_screen()
        preprocessed = self.preprocess_image(screenshot)
        text = self.extract_text(preprocessed)
        cleaned_text = self.clean_text(text)
        
        self.last_capture_time = time.time()
        
        return {
            "text": cleaned_text,
            "window_title": window_title,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# ===== AI CLASSIFICATION MODULE =====

class ActivityClassifier:
    """Classifies screen activity using Meta-Llama-3-2-3B-Instruct via Akash API."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = os.getenv("AKASH_API_KEY")
        if not self.api_key:
            logger.warning("AKASH_API_KEY not found in environment variables!")
        
        self.api_base_url = "https://chatapi.akash.network/api/v1"
        self.timeout = config.get("api_timeout", 10)
        
        # Enhanced prompt template that focuses on context, not content
        self.prompt_template = textwrap.dedent(""""
            Analyze this screen activity and classify it as exactly one of: 
            [STUDY] | [SOCIAL_MEDIA] | [NEUTRAL] | [WORK] | [ENTERTAINMENT] | [GAMES]
            
            IMPORTANT CLASSIFICATION RULES:
            - Focus on the CONTEXT and PLATFORM, not the specific content being displayed
            - If it's a known entertainment platform (Netflix, YouTube, etc.), classify as ENTERTAINMENT regardless of what shows/movies are displayed
            - If it's a known social media platform (Instagram, Facebook, etc.), classify as SOCIAL_MEDIA regardless of posts content
            - If it's a game launcher or gaming platform, classify as GAMES
            - STUDY: Educational platforms, research tools, learning websites, academic content
            - WORK: Coding IDEs, office software, productivity tools, business applications
            - NEUTRAL: Email, calendar, maps, system tools, general browsing
            
            CONTEXT CLUES:
            Window Title: "{window_title}"
            Content Preview: "{text}"
            
            Return ONLY the classification label without any explanation or additional text.
        """)
        self.classify_with_cache = functools.lru_cache(maxsize=100)(self._classify_activity)
    
    def _classify_activity(self, text: str, window_title: str) -> str:
        """Send text to Meta-Llama-3 via Akash API with enhanced context."""
        if not text.strip() and not window_title.strip():
            return "NEUTRAL"
        
        try:
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base_url
            )
            
            response = client.chat.completions.create(
                model="Qwen3-235B-A22B-Instruct-2507-FP8",
                messages=[
                    {
                        "role": "user",
                        "content": self.prompt_template.format(
                            text=text[:200],  # Limit text to avoid confusion
                            window_title=window_title
                        )
                    }
                ],
                temperature=0.1,  # Lower temperature for more consistent results
                max_tokens=10,
            )
            
            label = response.choices[0].message.content.strip()
            # Extract label if wrapped in brackets (e.g., "[WORK]")
            if "[" in label and "]" in label:
                label = re.search(r'\[(.*?)\]', label).group(1)
            return label.upper()  # Ensure consistent casing
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return "NEUTRAL"  # Default to neutral on error
    
    def classify_activity(self, window_data: Dict[str, str]) -> str:
        """Enhanced classification prioritizing window context over content."""
        text = window_data.get("text", "")
        window_title = window_data.get("window_title", "").lower()
        text_lower = text.lower()
        
        # Combined analysis with window title having higher priority
        combined_context = f"{window_title} {text_lower}"
        
        # ===== STRICT PLATFORM DETECTION FIRST =====
        
        # Game detection (highest priority - most strict)
        game_platforms = [
            "steam", "epic games", "epicgames", "battle.net", "battlenet", "origin", 
            "uplay", "ubisoft", "gog galaxy", "xbox", "playstation", "riot client",
            "fortnite", "minecraft", "league of legends", "valorant", "overwatch",
            "call of duty", "apex legends", "counter-strike", "dota", "rocket league"
        ]
        
        if any(platform in window_title for platform in game_platforms):
            return "GAMES"
        
        # Entertainment platform detection (high priority)
        entertainment_platforms = {
            "netflix": "netflix",
            "youtube": "youtube.com", 
            "hulu": "hulu.com",
            "disney+": "disneyplus",
            "prime video": "prime video",
            "hbomax": "hbomax",
            "hbo max": "hbo max",
            "crunchyroll": "crunchyroll",
            "twitch": "twitch.tv",
            "spotify": "spotify",
            "apple tv": "apple tv+",
            "paramount+": "paramount+"
        }
        
        for platform_name, pattern in entertainment_platforms.items():
            if pattern in window_title:
                return "ENTERTAINMENT"
        
        # Social media detection (high priority)
        social_platforms = {
            "instagram": "instagram.com",
            "facebook": "facebook.com", 
            "twitter": "twitter.com",
            "tiktok": "tiktok.com",
            "reddit": "reddit.com",
            "linkedin": "linkedin.com",  # Could be work-related, but often distraction
            "pinterest": "pinterest.com",
            "snapchat": "snapchat.com",
            "whatsapp": "whatsapp.com"
        }
        
        for platform_name, pattern in social_platforms.items():
            if pattern in window_title:
                if platform_name == "linkedin":
                    # LinkedIn could be work-related, need deeper analysis
                    break
                return "SOCIAL_MEDIA"
        
        # Work/Study platform detection
        work_platforms = [
            "visual studio code", "vscode", "pycharm", "intellij", "sublime text",
            "terminal", "cmd.exe", "powershell", "command prompt",
            "microsoft excel", "microsoft word", "microsoft powerpoint",
            "google docs", "google sheets", "google slides",
            "overleaf", "jupyter", "rstudio", "matlab"
        ]
        
        study_platforms = [
            "coursera", "edx", "khan academy", "udemy", "udacity", "codecademy",
            "quizlet", "anki", "researchgate", "jstor", "arxiv", "scholar"
        ]
        
        if any(platform in window_title for platform in work_platforms):
            return "WORK"
        
        if any(platform in window_title for platform in study_platforms):
            return "STUDY"
        
        # ===== CONTENT-BASED DETECTION (with safeguards) =====
        
        # Enhanced entertainment detection in content (with context awareness)
        entertainment_indicators = [
            "watch now", "play episode", "season", "episode", "trailer", 
            "browse titles", "continue watching", "my list", "new episodes",
            "trending now", "popular on netflix", "top picks for you"
        ]
        
        # Only classify as entertainment if we have strong platform context
        platform_context = any(ctx in window_title for ctx in ["netflix", "youtube", "hulu", "disney", "prime video"])
        entertainment_content = any(indicator in text_lower for indicator in entertainment_indicators)
        
        if platform_context and entertainment_content:
            return "ENTERTAINMENT"
        
        # Social media content patterns (with context)
        social_indicators = [
            "home", "stories", "reels", "feed", "timeline", "tweets", "posts",
            "followers", "following", "likes", "comments", "share", "retweet"
        ]
        
        social_context = any(ctx in window_title for ctx in ["instagram", "facebook", "twitter", "tiktok"])
        social_content = any(indicator in text_lower for indicator in social_indicators)
        
        if social_context and social_content:
            return "SOCIAL_MEDIA"
        
        # ===== FALLBACK TO AI CLASSIFICATION =====
        # Use both window title and limited text for AI classification
        classification_text = f"Window: {window_title}. Context: {text[:100]}"  # Limit text to avoid confusion
        
        ai_result = self.classify_with_cache(classification_text, window_title)
        
        # Post-process AI results for common misclassifications
        if ai_result == "GAMES":
            # Verify it's actually a game (avoid false positives from words like "play")
            if not any(game_indicator in window_title for game_indicator in ["game", "steam", "epic", "battle"]):
                return "ENTERTAINMENT"  # Probably entertainment, not games
        
        return ai_result
    
    def _extract_website(self, window_title: str) -> Optional[str]:
        """Enhanced website extraction focusing on domain patterns."""
        # Browser window title patterns
        browser_patterns = [
            r"([^ -]+ - (Google Chrome|Mozilla Firefox|Microsoft Edge|Safari))",
            r"(Google Chrome|Mozilla Firefox|Microsoft Edge|Safari)",
            r"(https?://[^\s/$.?#].[^\s]*)",
            r"([\w-]+\.(com|org|net|io|edu|gov))"
        ]
        
        for pattern in browser_patterns:
            match = re.search(pattern, window_title)
            if match:
                return match.group(1) if match.groups() else match.group(0)
        return None
    
    def _classify_website(self, url: str) -> str:
        """Enhanced website classification with platform priority."""
        url_lower = url.lower()
        
        # Priority 1: Entertainment platforms
        entertainment_domains = [
            "netflix.com", "youtube.com", "hulu.com", "disneyplus.com",
            "hbomax.com", "primevideo.com", "crunchyroll.com", "twitch.tv",
            "spotify.com", "appletv.com"
        ]
        
        if any(domain in url_lower for domain in entertainment_domains):
            return "ENTERTAINMENT"
        
        # Priority 2: Social media
        social_domains = [
            "instagram.com", "facebook.com", "twitter.com", "tiktok.com",
            "reddit.com", "pinterest.com", "snapchat.com"
        ]
        
        if any(domain in url_lower for domain in social_domains):
            return "SOCIAL_MEDIA"
        
        # Priority 3: Games
        game_domains = [
            "steampowered.com", "epicgames.com", "battle.net", "roblox.com",
            "minecraft.net", "leagueoflegends.com", "valorant.com"
        ]
        
        if any(domain in url_lower for domain in game_domains):
            return "GAMES"
        
        return "NEUTRAL"  # Default to neutral for unknown sites

# ===== INTERVENTION MODULE =====

class InterventionManager:
    """Manages intervention actions when distractions are detected."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.distraction_count = 0
        self.last_notification_time = 0
        self.notification_frequency = config.get("notification_frequency", 2) * 60  # convert to seconds
        
        # Check if voice alerts can be enabled
        self.voice_available = GTTS_AVAILABLE and PYDUB_AVAILABLE
        self.enable_voice_alerts = config.get("enable_voice_alerts", True) and self.voice_available
        
        if config.get("enable_voice_alerts", True) and not self.voice_available:
            logger.warning("Voice alerts requested but dependencies not available (gtts/pydub/ffmpeg)")
            logger.info("Voice alerts will be disabled. Install: pip install gtts pydub && install ffmpeg")
        
        self.enable_app_blocking = config.get("enable_app_blocking", True)
        self.temp_dir = "temp"
        
        # Create temp directory if it doesn't exist
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def handle_activity(self, activity_type: str, window_data: Dict[str, str]) -> Dict[str, Any]:
        """Process the detected activity and take appropriate action."""
        timestamp = window_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        result = {
            "timestamp": timestamp,
            "activity_type": activity_type,
            "content_preview": window_data.get("text", "")[:100] + "..." if len(window_data.get("text", "")) > 100 else window_data.get("text", ""),
            "window_title": window_data.get("window_title", ""),
            "action_taken": None
        }
        
        # Reset counter for productive activities
        if activity_type in ["STUDY", "WORK", "NEUTRAL"]:
            self.distraction_count = 0
            return result
        
        # Increment counter for distractions
        self.distraction_count += 1
        result["distraction_count"] = self.distraction_count
        
        # Take action if threshold reached
        if self.distraction_count >= self.config.get("distraction_threshold", 3):
            current_time = time.time()
            logger.info(f"Intervention triggered for {activity_type}. Count: {self.distraction_count}")
            
            # Always try to block distractions first
            distraction_blocked = False
            if self.enable_app_blocking:
                logger.info(f"Attempting to block {activity_type}")
                # Special handling for games
                if activity_type == "GAMES":
                    app_blocked = self.block_game(window_data)
                    if app_blocked:
                        logger.info(f"Successfully blocked game: {app_blocked}")
                        result["action_taken"] = "app_blocked"
                        result["app_blocked"] = app_blocked
                        distraction_blocked = True
                else:
                    app_blocked = self.block_distraction(activity_type, window_data)
                    if app_blocked:
                        logger.info(f"Successfully blocked distraction: {app_blocked}")
                        result["action_taken"] = "tab_closed"
                        result["app_blocked"] = app_blocked
                        distraction_blocked = True
            
            # ALWAYS send notification when distraction is successfully blocked
            if distraction_blocked:
                logger.info(f"Distraction blocked - sending notification for {activity_type}")
                self.send_notification(activity_type, window_data.get("window_title", ""), blocked=True)
                
                # Voice alert when distraction is blocked (if available)
                if self.enable_voice_alerts and activity_type:
                    self.play_voice_alert(activity_type, blocked=True)
                
                # Update notification time since we sent a notification
                self.last_notification_time = current_time
            else:
                # Only send regular notifications if frequency allows and no blocking occurred
                if current_time - self.last_notification_time >= self.notification_frequency:
                    logger.info(f"Sending regular notification for {activity_type}")
                    self.send_notification(activity_type, window_data.get("window_title", ""), blocked=False)
                    if not result.get("action_taken"):
                        result["action_taken"] = "notification"
                    
                    # Voice alert for regular notifications (if available)
                    if self.enable_voice_alerts and activity_type:
                        self.play_voice_alert(activity_type, blocked=False)
                        if not result.get("action_taken"):
                            result["action_taken"] = "voice_alert"
                    
                    # Update notification time
                    self.last_notification_time = current_time
            
            # Reset counter after intervention
            self.distraction_count = 0
        
        return result
    
    def block_game(self, window_data: Dict[str, str]) -> Optional[str]:
        """Close game launchers and applications."""
        if not self.enable_app_blocking:
            return None
            
        content = window_data.get("window_title", "") + " " + window_data.get("text", "")
        content_lower = content.lower()
        game_blocked = None
        
        # Enhanced game detection
        game_indicators = {
            'steam': 'Steam',
            'epic': 'Epic Games', 
            'epicgames': 'Epic Games',
            'epic games': 'Epic Games',
            'battle.net': 'Battle.net',
            'battlenet': 'Battle.net',
            'origin': 'Origin',
            'uplay': 'Uplay',
            'ubisoft': 'Ubisoft Connect',
            'gog': 'GOG Galaxy',
            'xbox': 'Xbox App',
            'fortnite': 'Fortnite',
            'minecraft': 'Minecraft',
            'league of legends': 'League of Legends',
            'valorant': 'Valorant',
            'overwatch': 'Overwatch'
        }

        # More flexible matching
        for keyword, description in game_indicators.items():
            if keyword in content_lower:
                game_blocked = description
                self._terminate_application(description)
                logger.info(f"Blocked game: {description}")
                return game_blocked
                
        # Additional check for browser games
        if any(x in content_lower for x in ['game', 'play', 'launcher']) and any(x in content_lower for x in ['.com', 'http']):
            self._close_specific_tab("gaming_website")
            return "Online Game"
            
        return None
    
    def send_notification(self, activity_type: str, content: str, blocked: bool = False) -> None:
        """Send system notification about distraction."""
        if not activity_type:
            return
            
        try:
            if blocked:
                message = f"✅ Blocked {activity_type.lower()} distraction"
                title = "🛡️ Focus Guard - Distraction Blocked"
            else:
                message = f"⚠️ Detected {activity_type.lower()} distraction"
                title = "⚠️ Focus Guard Alert"
            
            if content:
                message += f": {content[:50]}{'...' if len(content) > 50 else ''}"
            
            logger.info(f"Sending notification: {message}")
            
            if platform.system() == "Windows":
                try:
                    from plyer import notification
                    notification.notify(
                        title=title,
                        message=message,
                        timeout=5
                    )
                    return
                except ImportError:
                    subprocess.run(["msg", "*", f"{title}: {message}"], shell=True)
            elif platform.system() == "Darwin":
                subprocess.run(["osascript", "-e", f'display notification "{message}" with title "{title}"'])
            else:  # Linux
                subprocess.run(["notify-send", title, message])
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def play_voice_alert(self, activity_type: str, blocked: bool = False) -> None:
        """Play voice alert about distraction (if dependencies available)."""
        if not activity_type or not self.voice_available:
            return
            
        try:
            # Use the temp directory we created instead of system temp
            temp_filename = f"voice_alert_{int(time.time())}.mp3"
            mp3_path = os.path.join(self.temp_dir, temp_filename)
                
            try:
                # Generate different messages based on whether distraction was blocked
                if blocked:
                    message = f"Distraction blocked! {activity_type.lower()} has been closed"
                else:
                    message = f"Warning! {activity_type.lower()} distraction detected"
                
                # Generate TTS and save as MP3
                tts = gTTS(
                    text=message,
                    lang='en',
                    slow=False
                )
                tts.save(mp3_path)
                
                # Play audio in a separate thread
                def _play_audio():
                    try:
                        sound = AudioSegment.from_mp3(mp3_path)
                        play(sound)
                        logger.info(f"Voice alert played for {activity_type}")
                    except Exception as e:
                        logger.error(f"Audio playback failed: {e}")
                        # Fallback to system beep on Windows
                        if platform.system() == "Windows":
                            try:
                                import winsound
                                winsound.Beep(1000, 500)
                                logger.info("Played system beep as fallback")
                            except Exception:
                                pass
                    finally:
                        # Cleanup file after a delay
                        def cleanup_file():
                            time.sleep(2)  # Wait for playback to finish
                            if os.path.exists(mp3_path):
                                try:
                                    os.remove(mp3_path)
                                except Exception:
                                    pass
                        threading.Thread(target=cleanup_file, daemon=True).start()

                # Run playback in a separate thread
                threading.Thread(target=_play_audio, daemon=True).start()

            except Exception as e:
                # Cleanup on error
                if os.path.exists(mp3_path):
                    try:
                        os.remove(mp3_path)
                    except Exception:
                        pass
                raise e

        except Exception as e:
            logger.error(f"Voice alert system failed: {e}")
            # System beep as fallback
            if platform.system() == "Windows":
                try:
                    import winsound
                    winsound.Beep(1000, 500)
                    logger.info("Played system beep as voice alert fallback")
                except Exception:
                    pass
        
    def block_distraction(self, activity_type: str, window_data: Dict[str, str]) -> Optional[str]:
        """Close specific tabs based on detected content."""
        if not self.enable_app_blocking:
            return None
            
        content = window_data.get("window_title", "") + " " + window_data.get("text", "")
        content_lower = content.lower()
        window_title = window_data.get("window_title", "").lower()
        site_blocked = None
        
        # Map of distraction sites to identify specific tabs to close
        distraction_sites = {
            'instagram': 'Instagram',
            'facebook': 'Facebook', 
            'twitter': 'Twitter',
            'tiktok': 'TikTok',
            'reddit': 'Reddit',
            'youtube': 'YouTube',
            'netflix': 'Netflix',
            'twitch': 'Twitch',
            'hulu': 'Hulu',
            'disney': 'Disney+',
            'hbo': 'HBO Max',
            'prime video': 'Prime Video'
        }
        
        # Enhanced matching - check both window title and content
        for site_key, description in distraction_sites.items():
            if (site_key in window_title or 
                site_key in content_lower or
                f"{site_key}.com" in content_lower):
                site_blocked = description
                logger.info(f"Detected {description} in window: {window_title[:50]}...")
                self._close_specific_tab(site_key)
                return site_blocked
        
        # Check for browser-based entertainment patterns
        entertainment_patterns = [
            ('watch', 'Video Streaming'),
            ('stream', 'Video Streaming'), 
            ('episode', 'TV Show'),
            ('movie', 'Movie'),
            ('series', 'TV Series')
        ]
        
        # Only block if it's clearly entertainment context
        if any(browser in window_title for browser in ['chrome', 'firefox', 'edge', 'safari']):
            for pattern, description in entertainment_patterns:
                if pattern in content_lower and any(ent in content_lower for ent in ['netflix', 'youtube', 'hulu', 'disney', 'prime']):
                    site_blocked = description
                    logger.info(f"Detected {description} content in browser")
                    self._close_specific_tab("entertainment_content")
                    return site_blocked
                
        # Check for gaming platforms
        game_platforms = ['steam', 'epic games','epicgames','launcher', 'battle.net']
        for platform_name in game_platforms:
            if platform_name in content_lower:
                # For game launchers, we still close the application
                self._terminate_application(platform_name)
                return platform_name
            
        return None
    
    def _close_specific_tab(self, target_site: str) -> None:
        """Close a specific browser tab containing the target site."""
        try:
            logger.info(f"Attempting to close tab for {target_site}")
            
            if platform.system() == "Windows":
                # Method 1: Try to find and close specific browser windows
                try:
                    import win32gui
                    import win32con
                    
                    def enum_windows_callback(hwnd, windows):
                        if win32gui.IsWindowVisible(hwnd):
                            window_title = win32gui.GetWindowText(hwnd).lower()
                            if any(browser in window_title for browser in ['chrome', 'firefox', 'edge', 'safari']):
                                if target_site.lower() in window_title:
                                    windows.append(hwnd)
                        return True
                    
                    windows = []
                    win32gui.EnumWindows(enum_windows_callback, windows)
                    
                    if windows:
                        # Close the specific browser window/tab
                        for hwnd in windows:
                            win32gui.SetForegroundWindow(hwnd)
                            time.sleep(0.3)
                            pyautogui.hotkey('ctrl', 'w')
                            logger.info(f"Closed specific tab for {target_site}")
                            return
                            
                except ImportError:
                    logger.warning("win32gui not available, using fallback method")
                except Exception as e:
                    logger.warning(f"Windows-specific tab closing failed: {e}")
            
            # Fallback method: Close current active tab
            # Get current window and close tab
            time.sleep(0.1)
            if platform.system() == "Windows":
                pyautogui.hotkey('ctrl', 'w')
            elif platform.system() == "Darwin":
                pyautogui.hotkey('command', 'w') 
            else:
                pyautogui.hotkey('ctrl', 'w')
                
            logger.info(f"Closed current tab (fallback method for {target_site})")
            time.sleep(0.5)  # Wait for tab to close
            
        except Exception as e:
            logger.error(f"Failed to close tab for {target_site}: {e}")
            # Last resort: try Alt+F4 for the entire window
            try:
                if platform.system() == "Windows":
                    pyautogui.hotkey('alt', 'f4')
                    logger.info(f"Closed entire window as fallback for {target_site}")
            except Exception as e2:
                logger.error(f"All tab closing methods failed: {e2}")
    
    def _terminate_application(self, app_name: str) -> None:
        """Terminate the specified application type."""
        try:
            app_name_lower = app_name.lower()
            
            if platform.system() == "Windows":
                processes = []
                
                if 'steam' in app_name_lower:
                    processes = ["steam.exe", "steamwebhelper.exe"]
                elif 'epic' in app_name_lower:
                    processes = ["epicgameslauncher.exe", "epicgameslauncher-worker.exe"]
                elif 'battle' in app_name_lower:
                    processes = ["battle.net.exe", "battle.net helper.exe"]
                elif 'game' in app_name_lower:
                    # Generic game process termination
                    processes = [".exe"]  # This won't work - need specific names
                    
                for process in processes:
                    try:
                        subprocess.run(["taskkill", "/F", "/IM", process], 
                                     stdout=subprocess.DEVNULL, 
                                     stderr=subprocess.DEVNULL,
                                     timeout=5)
                        logger.info(f"Terminated {process}")
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Timeout terminating {process}")
                    except Exception as e:
                        logger.debug(f"Could not terminate {process}: {e}")
                        
            # Similar improvements for macOS and Linux...
            
        except Exception as e:
            logger.error(f"Error terminating {app_name}: {e}")
    
    def update_settings(self, config: Dict[str, Any]) -> None:
        """Update intervention settings from config."""
        self.config = config
        self.notification_frequency = config.get("notification_frequency", 2) * 60  # convert to seconds
        
        # Check if voice alerts can be enabled
        self.enable_voice_alerts = config.get("enable_voice_alerts", True) and self.voice_available
        self.enable_app_blocking = config.get("enable_app_blocking", True)

# ===== DATA MANAGEMENT MODULE =====

class ActivityTracker:
    """Tracks and stores activity history with privacy controls."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.activities = []
        self.max_history = 1000  # Maximum activities to keep in memory
        self.privacy_settings = config.get("privacy", {})
        self.data_file = "focus_data.json"
    
    def add_activity(self, activity_data: Dict[str, Any]) -> None:
        """Add an activity entry to tracking history."""
        # Apply privacy controls
        if self.privacy_settings.get("anonymize_personal_data", True):
            activity_data = self._anonymize_data(activity_data)
            
        # Store only what's allowed by privacy settings
        if not self.privacy_settings.get("store_raw_text", False):
            if "content_preview" in activity_data:
                # Keep only first few words as a preview
                words = activity_data["content_preview"].split()
                activity_data["content_preview"] = " ".join(words[:5]) + "..." if len(words) > 5 else activity_data["content_preview"]
        
        # Add to history
        self.activities.append(activity_data)
        
        # Trim history if needed
        if len(self.activities) > self.max_history:
            self.activities = self.activities[-self.max_history:]
    
    def _anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize potentially sensitive information."""
        anonymized = data.copy()
        
        if "content_preview" in anonymized:
            # Replace emails
            anonymized["content_preview"] = re.sub(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "[EMAIL]",
                anonymized["content_preview"]
            )
            
            # Replace phone numbers
            anonymized["content_preview"] = re.sub(
                r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
                "[PHONE]",
                anonymized["content_preview"]
            )
            
            # Replace URLs
            anonymized["content_preview"] = re.sub(
                r'https?://\S+',
                "[URL]",
                anonymized["content_preview"]
            )
        
        return anonymized
    
    def get_recent_activities(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent activity history."""
        return self.activities[-count:] if self.activities else []
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics from activity history."""
        if not self.activities:
            return {
                "total_tracked_time": 0,
                "activity_breakdown": {},
                "intervention_count": 0,
                "productivity_score": 0
            }
        
        # Count activity types
        activity_counts = {}
        intervention_count = 0
        
        for activity in self.activities:
            activity_type = activity.get("activity_type", "UNKNOWN")
            activity_counts[activity_type] = activity_counts.get(activity_type, 0) + 1
            
            if activity.get("action_taken"):
                intervention_count += 1
        
        # Calculate total tracked time (rough estimate)
        total_time_minutes = len(self.activities) * (self.config.get("screenshot_interval", 8) / 60)
        
        # Calculate productivity score (0-100)
        productive_types = ["STUDY", "WORK", "NEUTRAL"]
        productive_count = sum(activity_counts.get(t, 0) for t in productive_types)
        productivity_score = int((productive_count / len(self.activities)) * 100) if self.activities else 0
        
        return {
            "total_tracked_time": total_time_minutes,
            "activity_breakdown": activity_counts,
            "intervention_count": intervention_count,
            "productivity_score": productivity_score
        }
    
    def save_data(self) -> None:
        """Save activity data to file if permitted by privacy settings."""
        if not self.privacy_settings.get("store_raw_text", False):
            # Save only summary data
            summary = self.get_summary_stats()
            try:
                with open(self.data_file, 'w') as f:
                    json.dump(summary, f)
            except Exception as e:
                logger.error(f"Failed to save data: {str(e)}")
        else:
            # Save full activity data
            try:
                with open(self.data_file, 'w') as f:
                    json.dump(self.activities, f)
            except Exception as e:
                logger.error(f"Failed to save data: {str(e)}")
    
    def load_data(self) -> None:
        """Load saved activity data if available."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.activities = data
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")

# ===== MAIN APPLICATION =====

class FocusGuard:
    """Main application class that coordinates all modules."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEFAULT_CONFIG
        self.screen_monitor = ScreenMonitor(self.config)
        self.activity_classifier = ActivityClassifier(self.config)
        self.intervention_manager = InterventionManager(self.config)
        self.activity_tracker = ActivityTracker(self.config)
        
        self.running = False
        self.monitoring_thread = None
        self.event_queue = queue.Queue()
    
    def start_monitoring(self) -> None:
        """Start the monitoring process in a separate thread."""
        if self.running:
            logger.warning("Monitoring already running")
            return
            
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Focus monitoring started")
        logger.info(f"Voice alerts: {'enabled' if self.intervention_manager.enable_voice_alerts else 'disabled (dependencies missing)'}")
        logger.info(f"App blocking: {'enabled' if self.intervention_manager.enable_app_blocking else 'disabled'}")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring process."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        
        # Save activity data before exiting
        self.activity_tracker.save_data()
        logger.info("Focus monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        while self.running:
            try:
                # Get window info
                window_data = self.screen_monitor.get_active_window_info()
                
                # Classify activity with the full window data
                activity_type = self.activity_classifier.classify_activity(window_data)
                
                # Handle the detected activity with full context
                result = self.intervention_manager.handle_activity(activity_type, window_data)
                
                # Track the activity
                self.activity_tracker.add_activity(result)
                
                # Add to event queue
                self.event_queue.put(result)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(5)
    
    def get_events(self) -> List[Dict[str, Any]]:
        """Get new events from the queue (non-blocking)."""
        events = []
        while not self.event_queue.empty():
            try:
                events.append(self.event_queue.get_nowait())
            except queue.Empty:
                break
        return events
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return self.activity_tracker.get_summary_stats()
    
    def get_recent_activities(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent activity history."""
        return self.activity_tracker.get_recent_activities(count)
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration settings."""
        self.config.update(new_config)
        
        # Update component configs
        self.screen_monitor.config.update(new_config)
        self.activity_classifier.config.update(new_config)
        
        # Update intervention manager
        self.intervention_manager.config.update(new_config)
        self.intervention_manager.enable_app_blocking = new_config.get("enable_app_blocking", True)
        self.intervention_manager.enable_voice_alerts = new_config.get("enable_voice_alerts", True) and self.intervention_manager.voice_available
        self.intervention_manager.notification_frequency = new_config.get("notification_frequency", 2) * 60  # convert to seconds
        
        # Update activity tracker settings
        self.activity_tracker.privacy_settings = new_config.get("privacy", {})
        
        logger.info("Configuration updated")