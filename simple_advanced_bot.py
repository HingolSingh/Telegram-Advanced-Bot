#!/usr/bin/env python3
"""
Advanced Multi-AI Telegram Bot.
Integrates with advanced_handlers.py and multi_ai_client.py.
"""
import logging
import os
import asyncio
from telegram.ext import Application
from telegram.constants import ParseMode
from dotenv import load_dotenv
from advanced_handlers import setup_handlers
from analytics import AnalyticsManager
from keep_alive import keep_alive

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=getattr(
        logging,
        os.getenv(
            'LOG_LEVEL',
            'INFO').upper(),
        logging.INFO),
    handlers=[
        logging.FileHandler('advanced_bot.log'),
        logging.StreamHandler()])
logger = logging.getLogger(__name__)


async def main():
    """Start the advanced Telegram bot."""
    # Validate required environment variables
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable is required")
        raise ValueError("TELEGRAM_BOT_TOKEN not set")

    # Initialize analytics manager
    analytics_manager = AnalyticsManager()

    # Create the Application
    application = Application.builder().token(telegram_bot_token).build()

    # Register handlers from advanced_handlers.py
    setup_handlers(application, analytics_manager)

    # Start keep_alive in a separate thread for Replit
    keep_alive()

    # Start the bot
    logger.info("Starting Advanced Telegram Bot...")
    await application.initialize()
    await application.start()
    await application.updater.start_polling(allowed_updates=["message", "callback_query"])

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        raise
