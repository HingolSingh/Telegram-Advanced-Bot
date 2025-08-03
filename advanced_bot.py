#!/usr/bin/env python3
"""
Advanced Multi-AI Telegram Bot with Webhook (Replit Ready).
"""

import logging
import os
import asyncio
import threading
from flask import Flask, request
from telegram import Update
from telegram.ext import Application
from telegram.constants import ParseMode
from dotenv import load_dotenv

from advanced_handlers import setup_handlers
from analytics import AnalyticsManager
from database import init_database

# Load environment variables
load_dotenv()

# Flask app for webhook
app = Flask(__name__)

# Logging configuration
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper(), logging.INFO),
    handlers=[logging.FileHandler('bot.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Env variables
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # e.g., https://your-url.repl.co
PORT = int(os.getenv("PORT", 5000))  # Default to 5000 if not set

# Global Application instance
application = Application.builder().token(BOT_TOKEN).build()
asyncio.run(application.initialize())  # Initialize the application

# Setup handlers and DB
analytics_manager = AnalyticsManager()
setup_handlers(application, analytics_manager)
asyncio.run(init_database())

# Webhook set function
async def set_webhook():
    try:
        await application.bot.set_webhook(f"{WEBHOOK_URL}/webhook")
        logger.info("Webhook set to %s/webhook", WEBHOOK_URL)
    except Exception as e:
        logger.error("Error setting webhook: %s", e)

# GET route for health check
@app.route("/", methods=["GET"])
def home():
    return "🤖 Telegram Webhook Bot is live!"

# POST route to receive updates
@app.route("/webhook", methods=["POST"])
def receive_update():
    logger.info("Received webhook update: %s", request.get_json())
    try:
        data = request.get_json(force=True)
        update = Update.de_json(data, application.bot)
        # Use a new event loop for synchronous processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(application.process_update(update))
        loop.close()
        logger.info("Webhook update processed successfully")
        return "OK"
    except Exception as e:
        logger.error("Error processing update: %s", str(e))
        return "Error", 500

# Run Flask App
def run_flask():
    app.run(host="0.0.0.0", port=PORT)

# Run both Flask and set webhook
if __name__ == "__main__":
    if BOT_TOKEN is None or WEBHOOK_URL is None:
        logger.error("BOT_TOKEN or WEBHOOK_URL not set")
        exit(1)
    asyncio.run(set_webhook())
    threading.Thread(target=run_flask).start()