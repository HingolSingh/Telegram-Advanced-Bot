"""
Configuration settings for the Telegram bot.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# AI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Bot Configuration
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "4000"))
CONVERSATION_TIMEOUT = int(
    os.getenv(
        "CONVERSATION_TIMEOUT",
        "3600"))  # 1 hour in seconds

# Rate Limiting Configuration
RATE_LIMIT_MESSAGES = int(
    os.getenv(
        "RATE_LIMIT_MESSAGES",
        "10"))  # messages per time window
RATE_LIMIT_WINDOW = int(
    os.getenv(
        "RATE_LIMIT_WINDOW",
        "60"))  # time window in seconds

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Bot Personality Configuration
BOT_SYSTEM_PROMPT = os.getenv(
    "BOT_SYSTEM_PROMPT",
    "You are a helpful, friendly, and engaging conversational AI assistant. "
    "Provide thoughtful, informative responses while maintaining a warm and approachable tone. "
    "Keep responses concise but comprehensive. If you're unsure about something, admit it honestly."
)
DATABASE_URL = os.getenv("DATABASE_URL")
