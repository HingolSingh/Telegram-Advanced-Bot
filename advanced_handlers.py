#!/usr/bin/env python3
"""
Advanced message handlers for the multi-AI Telegram bot.
Supports Gemini (default), Grok xAI, OpenAI, Anthropic, Stability AI, Cohere, Together.ai, DeepInfra, Hugging Face, OpenRouter, and Groq Mixtral.
Features: Smart commands, voice, image processing, file handling, analytics, education, web search, translation, weather, code assistant, quizzes, reminders, summarization, math solving, learning assistant.
"""
import aiofiles  # Add this to the top of advanced_handlers.py if not already present
import logging
import asyncio
import os
import io
import tempfile
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import aiohttp
import re

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,  # lowercase filters required for v20+
)
from telegram.constants import ChatAction, ParseMode

from multi_ai_client import MultiAIClient
from database import (
    get_or_create_user,
    save_conversation,
    get_user_memory,
    save_user_memory,
    delete_user_memory,
)

from analytics import AnalyticsManager
from file_processor import FileProcessor
from educational_assistant import EducationalAssistant
from rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class AdvancedHandlers:
    """Advanced handlers for comprehensive Telegram bot functionality."""

    def __init__(self, analytics_manager: AnalyticsManager):
        self.ai_client = MultiAIClient()
        self.analytics = analytics_manager
        self.file_processor = FileProcessor()
        self.edu_assistant = EducationalAssistant(self.ai_client)
        self.rate_limiter = RateLimiter(max_requests=10, window_seconds=60)
        self.user_contexts = {}
        self.personality_modes = {}
        # Use env variable for admin users
        self.admin_users = {int(os.getenv("ADMIN_USERS", "123456789"))}
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.weather_api_key = os.getenv("OPENWEATHER_API_KEY")
        self.max_context_length = 20

    async def start_command(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command with user registration and feature overview."""
        user = update.effective_user
        user_id = user.id
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly. Please wait a moment.")
            return
        db_user = await get_or_create_user(
            telegram_id=user_id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            preferred_ai="gemini",
        )
        self.user_contexts[user_id] = [
            {"role": "system", "content": "You are a helpful and friendly AI assistant."}]
        context.user_data["model"] = "gemini"
        context.user_data["language"] = "en"
        context.user_data["private_mode"] = False
        self.personality_modes[user_id] = "friendly"
        await self.analytics.track_event(user_id, "start_command", {"user_id": user_id})
        models = self.ai_client.get_available_models()
        model_text = "\n".join(
            [f"• {info['name']} ({'Free' if info['free'] else 'Paid'})" for key, info in models.items()])
        welcome_message = f"""
🤖 Welcome to Your Advanced AI Assistant, {user.first_name}!
I'm a versatile bot powered by multiple AI providers, offering a wide range of capabilities.

🧠 **Supported AI Models**:
{model_text}

📚 **Commands**:
/help, /profile, /settings, /ai, /ask, /image, /readimage, /transcribe, /file, /web, /news, /translate, /weather, /code, /quiz, /reminder, /summarize, /math, /learn, /memory, /private, /admin
"""
        await update.message.reply_text(welcome_message, parse_mode=ParseMode.MARKDOWN_V2)

    async def help_command(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command to display available commands."""
        user_id = update.effective_user.id
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly. Please wait.")
            return
        help_text = """
📚 **Available Commands**:
/start - Start the bot
/help - Show this help message
/profile - View your profile
/settings - Adjust bot settings
/ai - Switch AI model
/ask - Ask a question
/image - Generate an image
/readimage - Analyze an image
/transcribe - Transcribe audio
/file - Process a file
/web - Web search
/news - Get news updates
/translate - Translate text
/weather - Get weather updates
/code - Get code assistance
/quiz - Generate a quiz
/reminder - Set a reminder
/summarize - Summarize text
/math - Solve math problems
/learn - Personal learning assistant
/memory - Manage conversation memory
/private - Toggle private mode
/admin - Admin commands (stats, broadcast)
"""
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN_V2)
        await self.analytics.track_event(user_id, "help_command", {})

    async def callback_query_handler(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle callback queries from inline keyboards."""
        query = update.callback_query
        user_id = query.from_user.id
        data = query.data
        if not self.rate_limiter.is_allowed(user_id):
            await query.message.reply_text("⚠️ You're sending requests too quickly\\. Please wait\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        await query.answer()
        try:
            if data.startswith("ai_"):
                model_key = data[3:]
                models = self.ai_client.get_available_models()
                if model_key in models:
                    context.user_data["model"] = model_key
                    db_user = await get_or_create_user(telegram_id=user_id)
                    db_user['preferred_ai'] = model_key
                    await db_user.save()
                    await query.message.reply_text(
                        f"✅ Switched to {models[model_key]['name']}\\ model\\.",
                        parse_mode=ParseMode.MARKDOWN_V2
                    )
                    await self.analytics.track_event(user_id, "switch_ai", {"model": model_key})
                    logger.debug(
                        f"User {user_id} switched to AI model {model_key}")
            elif data.startswith("settings_"):
                if data == "settings_model":
                    await self.switch_ai_command(update, context)
                    logger.debug(
                        f"User {user_id} requested to change AI model via settings")
                elif data == "settings_language":
                    keyboard = [[InlineKeyboardButton(lang, callback_data=f"lang_{lang}")] for lang in [
                        "en", "hi", "es", "fr"]]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    await query.message.reply_text("🌐 Choose a language:", reply_markup=reply_markup)
                    logger.debug(
                        f"User {user_id} requested to change language")
                elif data == "settings_personality":
                    keyboard = [[InlineKeyboardButton(mode, callback_data=f"pers_{mode}")] for mode in [
                        "friendly", "professional", "funny"]]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    await query.message.reply_text("😊 Choose a personality:", reply_markup=reply_markup)
                    logger.debug(
                        f"User {user_id} requested to change personality")
            elif data.startswith("lang_"):
                lang = data[5:]
                context.user_data["language"] = lang
                await query.message.reply_text(f"✅ Language set to {lang}\\.", parse_mode=ParseMode.MARKDOWN_V2)
                await self.analytics.track_event(user_id, "change_language", {"language": lang})
                logger.debug(f"User {user_id} changed language to {lang}")
            elif data.startswith("pers_"):
                personality = data[5:]
                self.personality_modes[user_id] = personality
                self.user_contexts[user_id] = [
                    {"role": "system", "content": f"You are a {personality} AI assistant."}]
                await query.message.reply_text(f"✅ Personality set to {personality}\\.", parse_mode=ParseMode.MARKDOWN_V2)
                await self.analytics.track_event(user_id, "change_personality", {"personality": personality})
                logger.debug(
                    f"User {user_id} changed personality to {personality}")
        except Exception as e:
            logger.error(f"Callback query error for user {user_id}: {e}")
            await query.message.reply_text("😔 Something went wrong\\. Try again later\\.", parse_mode=ParseMode.MARKDOWN_V2)

  #  async def profile_command(
  #  self, update: Update,        context:                     ContextTypes.DEFAULT_TYPE) -> None:
 #   user = update.effective_user
   # full_name = user.full_name or "Unknown"
   # username = user.username or "N_A"
   # user_id = user.id
  #  language_code = user.language_code or "N/A"
   # join_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # MarkdownV2 requires special characters to be escaped
   # def escape_md(text: str) -> str:
     #   escape_chars = r"\_*[]()~`>#+-=|{}.!<>"
      #  return ''.join(['\\' + c if c in escape_chars else c for c in text])

 #   profile_text = (
  #      f"👤 *Your Profile*\n\n"
     #   f"*Name*: {escape_md(full_name)}\n"
     #   f"*Username*: @{escape_md(username)}\n"
      #  f"*Telegram ID*: `{user_id}`\n"
     #   f"*Language*: {escape_md(language_code)}\n"
      #  f"*Joined*: `{join_date}`"
 #   )

  #  if user.photo:
    #    photos = await context.bot.get_user_profile_photos(user.id)
     #   if photos.total_count > 0:
        #    photo_file = photos.photos[0][0].file_id
         #   await update.message.reply_photo(
          #      photo=photo_file,
                #caption=profile_text,
                #parse_mode="MarkdownV2"
      #      )
    #        return

  #  await update.message.reply_text(profile_text, parse_mode="MarkdownV2")

    async def image_command(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /image command to generate images."""
        user_id = update.effective_user.id
        logger.debug(f"User {user_id} requested image generation")
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly\\. Please wait\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        prompt = " ".join(context.args) if context.args else None
        if not prompt:
            await update.message.reply_text("🖼️ Please provide a description for the image \\(e\\.g\\., /image A futuristic city\\)\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        if len(prompt) > 500:  # Basic prompt length validation
            await update.message.reply_text("⚠️ Prompt is too long\\. Please keep it under 500 characters\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
            image_data = await self.ai_client.generate_image(str(user_id), prompt, model="stability")
            if image_data:
                await update.message.reply_photo(photo=image_data)
                await self.analytics.track_event(user_id, "image_command", {"prompt": prompt[:50]})
                logger.debug(
                    f"Image generated for user {user_id} with prompt: {prompt[:50]}")
            else:
                await update.message.reply_text("❌ Failed to generate image\\. Try a different prompt\\.", parse_mode=ParseMode.MARKDOWN_V2)
        except Exception as e:
            logger.error(f"Image generation error for user {user_id}: {e}")
            await update.message.reply_text(f"😔 Failed to generate image: {str(e)}\\.", parse_mode=ParseMode.MARKDOWN_V2)

    async def read_image_command(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /readimage command to analyze images."""
        user_id = update.effective_user.id
        logger.debug(f"User {user_id} requested image analysis")
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly\\. Please wait\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        if not update.message.photo:
            await update.message.reply_text("🖼️ Please send an image to analyze\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        try:
            photo = update.message.photo[-1]
            if photo.file_size > 20 * 1024 * \
                    1024:  # Check file size (20 MB limit)
                await update.message.reply_text("⚠️ Image is too large\\. Please send a smaller image\\.", parse_mode=ParseMode.MARKDOWN_V2)
                return
            file = await context.bot.get_file(photo.file_id)
            image_data = await file.download_as_bytearray()
            description = await self.ai_client.analyze_image(str(user_id), image_data, model="openai")
            await update.message.reply_text(f"🖼️ Image Description: {description}\\.", parse_mode=ParseMode.MARKDOWN_V2)
            await self.analytics.track_event(user_id, "read_image_command", {})
            logger.debug(f"Image analyzed for user {user_id}")
        except Exception as e:
            logger.error(f"Image analysis error for user {user_id}: {e}")
            await update.message.reply_text(f"😔 Failed to analyze image: {str(e)}\\.", parse_mode=ParseMode.MARKDOWN_V2)

    async def transcribe_command(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /transcribe command for audio transcription."""
        user_id = update.effective_user.id
        logger.debug(f"User {user_id} requested audio transcription")
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly\\. Please wait\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        if not update.message.voice:
            await update.message.reply_text("🎤 Please send a voice message to transcribe\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        try:
            voice_file = await context.bot.get_file(update.message.voice.file_id)
            if voice_file.file_size > 20 * 1024 * \
                    1024:  # Check file size (20 MB limit)
                await update.message.reply_text("⚠️ Voice message is too large\\. Please send a smaller file\\.", parse_mode=ParseMode.MARKDOWN_V2)
                return
            audio_data = await voice_file.download_as_bytearray()
            transcript = await self.ai_client.transcribe_audio(str(user_id), audio_data, "ogg")
            if transcript and "error" not in transcript.lower():
                await update.message.reply_text(f"🎤 **Transcription**: {transcript}\\.", parse_mode=ParseMode.MARKDOWN_V2)
                await self.analytics.track_event(user_id, "transcribe_command", {})
                logger.debug(f"Audio transcribed for user {user_id}")
            else:
                await update.message.reply_text("❌ Failed to transcribe audio\\. Try a different file\\.", parse_mode=ParseMode.MARKDOWN_V2)
        except Exception as e:
            logger.error(f"Audio transcription error for user {user_id}: {e}")
            await update.message.reply_text(f"😔 Failed to transcribe audio: {str(e)}\\.", parse_mode=ParseMode.MARKDOWN_V2)

    async def file_command(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /file command to process uploaded files."""
        user_id = update.effective_user.id
        logger.debug(f"User {user_id} requested file processing")
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly\\. Please wait\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        if not update.message.document:
            await update.message.reply_text("📄 Please upload a file to process\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        try:
            document = update.message.document
            if document.file_size > 20 * 1024 * \
                    1024:  # Check file size (20 MB limit)
                await update.message.reply_text("⚠️ File is too large\\. Please send a smaller file\\.", parse_mode=ParseMode.MARKDOWN_V2)
                return
            file = await context.bot.get_file(document.file_id)
            file_data = await file.download_as_bytearray()
            result = await self.file_processor.process_file(file_data, document.file_name)
            await update.message.reply_text(f"📄 File processed: {result}\\.", parse_mode=ParseMode.MARKDOWN_V2)
            await self.analytics.track_event(user_id, "file_command", {"file_type": document.file_name.split('.')[-1]})
            logger.debug(
                f"File processed for user {user_id}: {document.file_name}")
        except Exception as e:
            logger.error(f"File processing error for user {user_id}: {e}")
            await update.message.reply_text(f"😔 Failed to process file: {str(e)}\\.", parse_mode=ParseMode.MARKDOWN_V2)

    async def web_search_command(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /web command for web search with DeepSearch."""
        user_id = update.effective_user.id
        logger.debug(f"User {user_id} requested web search")
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly\\. Please wait\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        query = " ".join(context.args) if context.args else None
        if not query:
            await update.message.reply_text("🌐 Please provide a search query \\(e\\.g\\., /web latest AI news\\)\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        if len(query) > 500:  # Query length validation
            await update.message.reply_text("⚠️ Search query is too long\\. Please keep it under 500 characters\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            results = await self.ai_client.web_search(str(user_id), query, mode="deepsearch")
            response = "\n".join(
                [f"• [{r['title']}]({r['url']})" for r in results[:5]])
            await update.message.reply_text(f"🌐 **Search Results**:\n{response}", parse_mode=ParseMode.MARKDOWN_V2)
            await self.analytics.track_event(user_id, "web_search", {"query": query[:50]})
            logger.debug(
                f"Web search completed for user {user_id}: {query[:50]}")
        except Exception as e:
            logger.error(f"Web search error for user {user_id}: {e}")
            await update.message.reply_text(f"😔 Failed to search the web: {str(e)}\\.", parse_mode=ParseMode.MARKDOWN_V2)

    async def news_command(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /news command to fetch news updates."""
        user_id = update.effective_user.id
        logger.debug(f"User {user_id} requested news updates")
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly\\. Please wait\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        query = " ".join(context.args) if context.args else "latest"
        if len(query) > 500:  # Query length validation
            await update.message.reply_text("⚠️ News query is too long\\. Please keep it under 500 characters\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://newsapi.org/v2/everything?q={query}&apiKey={self.news_api_key}") as resp:
                    data = await resp.json()
                    articles = data.get("articles", [])[:5]
                    response = "\n".join(
                        [f"• [{a['title']}]({a['url']})" for a in articles])
                    await update.message.reply_text(f"📰 **News**:\n{response}", parse_mode=ParseMode.MARKDOWN_V2)
                    await self.analytics.track_event(user_id, "news_command", {"query": query[:50]})
                    logger.debug(
                        f"News fetched for user {user_id}: {query[:50]}")
        except Exception as e:
            logger.error(f"News fetch error for user {user_id}: {e}")
            await update.message.reply_text(f"😔 Failed to fetch news: {str(e)}\\.", parse_mode=ParseMode.MARKDOWN_V2)

    async def translate_command(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /translate command to translate text."""
        user_id = update.effective_user.id
        logger.debug(f"User {user_id} requested translation")
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly\\. Please wait\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        args = context.args
        if len(args) < 2:
            await update.message.reply_text("🌐 Usage: /translate <target_language> <text>", parse_mode=ParseMode.MARKDOWN_V2)
            return
        target_lang, text = args[0], " ".join(args[1:])
        if len(text) > 500:  # Text length validation
            await update.message.reply_text("⚠️ Text is too long\\. Please keep it under 500 characters\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        valid_langs = ["en", "hi", "es", "fr", "de",
                       "it", "ja", "zh"]  # Example ISO 639-1 codes
        if target_lang.lower() not in valid_langs:
            await update.message.reply_text(f"⚠️ Invalid language code\\. Try one of: {', '.join(valid_langs)}\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        try:
            translation = await self.ai_client.translate_text(str(user_id), text, target_lang)
            await update.message.reply_text(f"🌐 **Translation** \\({target_lang}\\): {translation}\\.", parse_mode=ParseMode.MARKDOWN_V2)
            await self.analytics.track_event(user_id, "translate_command", {"language": target_lang})
            logger.debug(
                f"Text translated for user {user_id} to {target_lang}")
        except Exception as e:
            logger.error(f"Translation error for user {user_id}: {e}")
            await update.message.reply_text(f"😔 Failed to translate text: {str(e)}\\.", parse_mode=ParseMode.MARKDOWN_V2)

    async def weather_command(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /weather command to fetch weather updates."""
        user_id = update.effective_user.id
        logger.debug(f"User {user_id} requested weather updates")
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly\\. Please wait\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        city = " ".join(context.args) if context.args else None
        if not city:
            await update.message.reply_text("🌤️ Please provide a city \\(e\\.g\\., /weather London\\)\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        if len(city) > 100:  # City query length validation
            await update.message.reply_text("⚠️ City name is too long\\. Please keep it under 100 characters\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.weather_api_key}&units=metric") as resp:
                    data = await resp.json()
                    if data.get("cod") != 200:
                        await update.message.reply_text("🌤️ City not found\\. Try again\\.", parse_mode=ParseMode.MARKDOWN_V2)
                        return
                    weather = data["weather"][0]["description"]
                    temp = data["main"]["temp"]
                    humidity = data["main"]["humidity"]
                    wind_speed = data["wind"]["speed"]
                    response = f"🌤️ **Weather in {city}**:\n" \
                               f"• Description: {weather}\n" \
                               f"• Temperature: {temp}°C\n" \
                               f"• Humidity: {humidity}%\n" \
                               f"• Wind Speed: {wind_speed} m/s"
                    await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN_V2)
                    await self.analytics.track_event(user_id, "weather_command", {"city": city})
                    logger.debug(f"Weather fetched for user {user_id}: {city}")
        except Exception as e:
            logger.error(f"Weather fetch error for user {user_id}: {e}")
            await update.message.reply_text(f"😔 Failed to fetch weather: {str(e)}\\.", parse_mode=ParseMode.MARKDOWN_V2)

    async def code_command(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /code command for code assistance."""
        user_id = update.effective_user.id
        logger.debug(f"User {user_id} requested code assistance")
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly\\. Please wait\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        query = " ".join(context.args) if context.args else None
        if not query:
            await update.message.reply_text("💻 Please provide a coding query \\(e\\.g\\., /code Write a Python function\\)\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        if len(query) > 500:  # Query length validation
            await update.message.reply_text("⚠️ Coding query is too long\\. Please keep it under 500 characters\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        try:
            response = await self.ai_client.generate_response(str(user_id), f"Provide code assistance: {query}", model="grok")
            await update.message.reply_text(f"💻 **Code Assistance**:\n```python\n{response}\n```", parse_mode=ParseMode.MARKDOWN_V2)
            await self.analytics.track_event(user_id, "code_command", {"query": query[:50]})
            logger.debug(
                f"Code assistance provided for user {user_id}: {query[:50]}")
        except Exception as e:
            logger.error(f"Code assistance error for user {user_id}: {e}")
            await update.message.reply_text(f"😔 Failed to provide code assistance: {str(e)}\\.", parse_mode=ParseMode.MARKDOWN_V2)

    async def quiz_command(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /quiz command to generate a quiz."""
        user_id = update.effective_user.id
        logger.debug(f"User {user_id} requested quiz generation")
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly\\. Please wait\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        topic = " ".join(context.args) if context.args else "general"
        if len(topic) > 100:  # Topic length validation
            await update.message.reply_text("⚠️ Quiz topic is too long\\. Please keep it under 100 characters\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        try:
            quiz = await self.edu_assistant.generate_quiz(str(user_id), topic)
            response = "\n".join(
                [
                    f"**{i+1}\\. {q['question']}**\n   Options: {', '.join(q['options'])}" for i,
                    q in enumerate(quiz)])
            await update.message.reply_text(f"🧠 **Quiz on {topic}**:\n{response}", parse_mode=ParseMode.MARKDOWN_V2)
            await self.analytics.track_event(user_id, "quiz_command", {"topic": topic})
            logger.debug(f"Quiz generated for user {user_id}: {topic}")
        except Exception as e:
            logger.error(f"Quiz generation error for user {user_id}: {e}")
            await update.message.reply_text(f"😔 Failed to generate quiz: {str(e)}\\.", parse_mode=ParseMode.MARKDOWN_V2)

    async def reminder_command(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /reminder command to set reminders."""
        user_id = update.effective_user.id
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly. Please wait.")
            return
        args = context.args
        if len(args) < 2:
            await update.message.reply_text("⏰ Usage: /reminder <time_in_minutes> <message>")
            return
        try:
            minutes = int(args[0])
            message = " ".join(args[1:])
            await asyncio.sleep(minutes * 60)
            await context.bot.send_message(chat_id=user_id, text=f"⏰ **Reminder**: {message}")
            await self.analytics.track_event(user_id, "reminder_command", {"message": message[:50]})
        except ValueError:
            await update.message.reply_text("⏰ Please provide a valid number of minutes.")
        except Exception as e:
            logger.error(f"Reminder error for user {user_id}: {e}")
            await update.message.reply_text("😔 Sorry, something went wrong while setting the reminder.")

    async def summarize_command(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /summarize command to summarize text."""
        user_id = update.effective_user.id
        logger.debug(f"User {user_id} requested text summarization")
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly\\. Please wait\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        text = " ".join(context.args) if context.args else None
        if not text:
            await update.message.reply_text("📝 Please provide text to summarize\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        if len(text) > 500:  # Text length validation
            await update.message.reply_text("⚠️ Text is too long\\. Please keep it under 500 characters\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        try:
            summary = await self.ai_client.generate_response(str(user_id), f"Summarize: {text}", model="gemini")
            await update.message.reply_text(f"📝 **Summary**:\n{summary}\\.", parse_mode=ParseMode.MARKDOWN_V2)
            await self.analytics.track_event(user_id, "summarize_command", {"text_length": len(text)})
            logger.debug(f"Text summarized for user {user_id}: {text[:50]}")
        except Exception as e:
            logger.error(f"Summarization error for user {user_id}: {e}")
            await update.message.reply_text(f"😔 Failed to summarize text: {str(e)}\\.", parse_mode=ParseMode.MARKDOWN_V2)

    async def math_command(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /math command to solve math problems."""
        user_id = update.effective_user.id
        logger.debug(f"User {user_id} requested math problem solving")
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly\\. Please wait\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        problem = " ".join(context.args) if context.args else None
        if not problem:
            await update.message.reply_text("➕ Please provide a math problem \\(e\\.g\\., /math 2x + 3 = 7\\)\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        if len(problem) > 500:  # Problem length validation
            await update.message.reply_text("⚠️ Math problem is too long\\. Please keep it under 500 characters\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        try:
            solution = await self.ai_client.generate_response(str(user_id), f"Solve math problem: {problem}", model="gemini")
            await update.message.reply_text(f"➕ **Solution**:\n```math\n{solution}\n```", parse_mode=ParseMode.MARKDOWN_V2)
            await self.analytics.track_event(user_id, "math_command", {"problem": problem[:50]})
            logger.debug(
                f"Math problem solved for user {user_id}: {problem[:50]}")
        except Exception as e:
            logger.error(f"Math solving error for user {user_id}: {e}")
            await update.message.reply_text(f"😔 Failed to solve math problem: {str(e)}\\.", parse_mode=ParseMode.MARKDOWN_V2)

    async def memory_command(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /memory command to manage conversation memory."""
        user_id = update.effective_user.id
        logger.debug(f"User {user_id} requested memory command")
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly\\. Please wait\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        args = context.args
        try:
            if not args:
                memory = await get_user_memory(user_id)
                response = "\n".join(
                    [f"**{m['timestamp']}**: {m['content']}" for m in memory]) if memory else "No memory found\\."
                await update.message.reply_text(f"🧠 **Conversation Memory**:\n{response}", parse_mode=ParseMode.MARKDOWN_V2)
                await self.analytics.track_event(user_id, "memory_command", {"action": "view"})
            elif args[0].lower() == "clear":
                await delete_user_memory(user_id)
                self.user_contexts[user_id] = [
                    {"role": "system", "content": f"You are a {self.personality_modes.get(user_id, 'friendly')} AI assistant."}]
                await update.message.reply_text("🧠 Conversation memory cleared\\.", parse_mode=ParseMode.MARKDOWN_V2)
                await self.analytics.track_event(user_id, "memory_command", {"action": "clear"})
            elif args[0].lower() == "export":
                memory = await get_user_memory(user_id)
                if memory:
                    async with aiofiles.tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp:
                        await tmp.write(json.dumps(memory, indent=2))
                        tmp_name = tmp.name
                    async with aiofiles.open(tmp_name, "rb") as f:
                        await update.message.reply_document(document=f, filename="memory_export.json")
                    os.unlink(tmp_name)
                    await self.analytics.track_event(user_id, "memory_command", {"action": "export"})
                else:
                    await update.message.reply_text("🧠 No memory to export\\.", parse_mode=ParseMode.MARKDOWN_V2)
                    await self.analytics.track_event(user_id, "memory_command", {"action": "export_empty"})
            else:
                await update.message.reply_text("⚠️ Invalid action\\. Use /memory, /memory clear, or /memory export\\.", parse_mode=ParseMode.MARKDOWN_V2)
                await self.analytics.track_event(user_id, "memory_command", {"action": "invalid"})
            logger.debug(
                f"Memory command completed for user {user_id}: {args[0] if args else 'view'}")
        except Exception as e:
            logger.error(f"Memory command error for user {user_id}: {e}")
            await update.message.reply_text(f"😔 Failed to process memory command: {str(e)}\\.", parse_mode=ParseMode.MARKDOWN_V2)

    async def learn_command(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /learn command for personal learning assistance."""
        user_id = update.effective_user.id
        logger.debug(f"User {user_id} requested learning content")
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly\\. Please wait\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        topic = " ".join(context.args) if context.args else None
        if not topic:
            await update.message.reply_text("📚 Please provide a topic to learn about \\(e\\.g\\., /learn Python programming\\)\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        if len(topic) > 100:  # Topic length validation
            await update.message.reply_text("⚠️ Topic is too long\\. Please keep it under 100 characters\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        try:
            content = await self.edu_assistant.generate_learning_content(str(user_id), topic)
            await update.message.reply_text(f"📚 **Learning Content on {topic}**:\n{content}\\.", parse_mode=ParseMode.MARKDOWN_V2)
            await self.analytics.track_event(user_id, "learn_command", {"topic": topic[:50]})
            logger.debug(
                f"Learning content generated for user {user_id}: {topic[:50]}")
        except Exception as e:
            logger.error(f"Learning content error for user {user_id}: {e}")
            await update.message.reply_text(f"😔 Failed to generate learning content: {str(e)}\\.", parse_mode=ParseMode.MARKDOWN_V2)

    async def private_command(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /private command to toggle private mode."""
        user_id = update.effective_user.id
        logger.debug(f"User {user_id} requested private mode toggle")
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly\\. Please wait\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        try:
            context.user_data["private_mode"] = not context.user_data.get(
                "private_mode", False)
            status = "enabled" if context.user_data["private_mode"] else "disabled"
            await update.message.reply_text(f"🔒 Private mode {status}\\.", parse_mode=ParseMode.MARKDOWN_V2)
            await self.analytics.track_event(user_id, "private_command", {"status": status})
            logger.debug(f"Private mode set to {status} for user {user_id}")
        except Exception as e:
            logger.error(f"Private command error for user {user_id}: {e}")
            await update.message.reply_text(f"😔 Failed to toggle private mode: {str(e)}\\.", parse_mode=ParseMode.MARKDOWN_V2)

    async def admin_command(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /admin command for admin functions."""
        user_id = update.effective_user.id
        logger.debug(f"User {user_id} requested admin command")
        if user_id not in self.admin_users:
            await update.message.reply_text("🔐 You are not authorized to use admin commands\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        args = context.args
        if not args:
            await update.message.reply_text(
                "🔐 **Admin Commands**\n\n"
                "Usage:\n• /admin stats - View bot usage stats\n• /admin broadcast <message> - Send message to all users",
                parse_mode=ParseMode.MARKDOWN_V2
            )
            return
        action = args[0].lower()
        try:
            if action == "stats":
                stats = await self.analytics.get_bot_stats()
                stats_text = f"""
🔧 **Bot Statistics**
• Total Users: {stats.get('total_users', 0)}
• Active Users \\(Last 24h\\): {stats.get('active_users', 0)}
• Total Messages: {stats.get('total_messages', 0)}
• AI Requests: {stats.get('ai_requests', 0)}
• Uptime: {stats.get('uptime', 'N/A')}
"""
                await update.message.reply_text(stats_text, parse_mode=ParseMode.MARKDOWN_V2)
                await self.analytics.track_event(user_id, "admin_stats", {})
                logger.debug(f"Stats retrieved for admin user {user_id}")
            elif action == "broadcast" and len(args) > 1:
                message = " ".join(args[1:])
                if len(message) > 500:  # Message length validation
                    await update.message.reply_text("⚠️ Broadcast message is too long\\. Please keep it under 500 characters\\.", parse_mode=ParseMode.MARKDOWN_V2)
                    return
                users = await self.analytics.get_all_users()
                success_count = 0
                for user in users:
                    try:
                        await context.bot.send_message(
                            chat_id=user["telegram_id"],
                            text=f"📢 **Admin Broadcast**: {message}",
                            parse_mode=ParseMode.MARKDOWN_V2
                        )
                        success_count += 1
                        await asyncio.sleep(0.1)  # Avoid rate limiting
                    except Exception as e:
                        logger.warning(
                            f"Failed to send broadcast to user {user['telegram_id']}: {e}")
                await update.message.reply_text(f"📢 Broadcast sent to {success_count}/{len(users)} users\\.", parse_mode=ParseMode.MARKDOWN_V2)
                await self.analytics.track_event(user_id, "admin_broadcast", {"message_length": len(message)})
                logger.debug(
                    f"Broadcast sent by user {user_id} to {success_count}/{len(users)} users")
            else:
                await update.message.reply_text("❓ Invalid admin command\\. Use /admin for usage\\.", parse_mode=ParseMode.MARKDOWN_V2)
        except Exception as e:
            logger.error(f"Admin command error for user {user_id}: {e}")
            await update.message.reply_text(f"😔 Failed to process admin command: {str(e)}\\.", parse_mode=ParseMode.MARKDOWN_V2)

    async def handle_text_message(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle plain text messages with intelligent routing."""
        user_id = update.effective_user.id
        logger.debug(f"User {user_id} sent text message")
        message_text = update.message.text
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly\\. Please wait\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        if len(message_text) > 1000:  # Message length validation
            await update.message.reply_text("⚠️ Message is too long\\. Please keep it under 1000 characters\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        await self._process_text_message(update, context, message_text)

    async def _process_text_message(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
            message_text: str) -> None:
        """Process text messages with context and personality."""
        user_id = update.effective_user.id
        logger.debug(
            f"Processing text message for user {user_id}: {message_text[:50]}")
        try:
            db_user = await get_or_create_user(
                telegram_id=user_id,
                username=update.effective_user.username,
                first_name=update.effective_user.first_name,
                last_name=update.effective_user.last_name
            )
            model = context.user_data.get(
                "model", db_user.preferred_ai or "gemini")
            personality = self.personality_modes.get(user_id, "friendly")
            if user_id not in self.user_contexts:
                self.user_contexts[user_id] = [
                    {"role": "system", "content": f"You are a {personality} AI assistant."}]
            self.user_contexts[user_id].append({
                "role": "user",
                "content": message_text,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            self.user_contexts[user_id] = self.user_contexts[user_id][-self.max_context_length:]
            try:
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            except Exception as e:
                logger.warning(
                    f"Failed to send chat action for user {user_id}: {e}")
            start_time = datetime.now()
            response = await self.ai_client.generate_response(
                str(user_id),
                message_text,
                model,
                conversation_history=self.user_contexts[user_id][:-1]
            )
            processing_time = (datetime.now() - start_time).total_seconds()
            self.user_contexts[user_id].append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            if not context.user_data.get("private_mode", False):
                await save_conversation(user_id, message_text, response, model, "text")
            await self.analytics.track_event(user_id, "text_message", {
                "model": model,
                "processing_time": processing_time,
                "message_length": len(message_text)
            })
            await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN_V2)
            logger.debug(
                f"Text message processed for user {user_id}: {response[:50]}")
        except Exception as e:
            logger.error(f"Text processing error for user {user_id}: {e}")
            await update.message.reply_text(f"😔 Failed to process your message: {str(e)}\\.", parse_mode=ParseMode.MARKDOWN_V2)

    async def handle_voice_message(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle voice messages by redirecting to /transcribe."""
        user_id = update.effective_user.id
        logger.debug(f"User {user_id} sent voice message")
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly\\. Please wait\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        try:
            await self.transcribe_command(update, context)
            logger.debug(f"Voice message processed for user {user_id}")
        except Exception as e:
            logger.error(
                f"Voice message processing error for user {user_id}: {e}")
            await update.message.reply_text(f"😔 Failed to process voice message: {str(e)}\\.", parse_mode=ParseMode.MARKDOWN_V2)

    async def handle_file_message(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle file uploads by redirecting to /file command."""
        user_id = update.effective_user.id
        logger.debug(f"User {user_id} sent file message")
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ You're sending messages too quickly\\. Please wait\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        try:
            await self.file_command(update, context)
            logger.debug(f"File message processed for user {user_id}")
        except Exception as e:
            logger.error(
                f"File message processing error for user {user_id}: {e}")
            await update.message.reply_text(f"😔 Failed to process file: {str(e)}\\.", parse_mode=ParseMode.MARKDOWN_V2)

    async def error_handler(
            self,
            update: object,
            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors gracefully."""
        logger.error(f"Bot error: {context.error}")
        if isinstance(update, Update) and update.message:
            await update.message.reply_text(f"🚨 An unexpected error occurred: {str(context.error)}\\.", parse_mode=ParseMode.MARKDOWN_V2)


def setup_handlers(application: Application,
                   analytics_manager: AnalyticsManager) -> None:
    """Register all handlers with the Telegram application."""
    logger.debug("Registering handlers with the Telegram application")
    try:
        handlers = AdvancedHandlers(analytics_manager)
        application.add_handler(
            CommandHandler(
                "start",
                handlers.start_command))
        application.add_handler(CommandHandler("help", handlers.help_command))
     #   application.add_handler(
      #      CommandHandler(
       #         "profile",
                #handlers.profile_command))
     #   application.add_handler(
      #      CommandHandler(
     #           "settings",
                #handlers.settings_command))
    #    application.add_handler(
    #        CommandHandler(
    #            "ai", handlers.switch_ai_command))
        application.add_handler(CommandHandler("ask", #handlers.ask_command))
    #    application.add_handler(
    #        CommandHandler(
     #           "image",
                handlers.image_command))
        application.add_handler(
            CommandHandler(
                "readimage",
                handlers.read_image_command))
        application.add_handler(
            CommandHandler(
                "transcribe",
                handlers.transcribe_command))
        application.add_handler(CommandHandler("file", handlers.file_command))
        application.add_handler(CommandHandler(
            "web", handlers.web_search_command))
        application.add_handler(CommandHandler("news", handlers.news_command))
        application.add_handler(
            CommandHandler(
                "translate",
                handlers.translate_command))
        application.add_handler(
            CommandHandler(
                "weather",
                handlers.weather_command))
        application.add_handler(CommandHandler("code", handlers.code_command))
        application.add_handler(CommandHandler("quiz", handlers.quiz_command))
        application.add_handler(
            CommandHandler(
                "reminder",
                handlers.reminder_command))
        application.add_handler(
            CommandHandler(
                "summarize",
                handlers.summarize_command))
        application.add_handler(CommandHandler("math", handlers.math_command))
        application.add_handler(
            CommandHandler(
                "memory",
                handlers.memory_command))
        application.add_handler(
            CommandHandler(
                "private",
                handlers.private_command))
        application.add_handler(
            CommandHandler(
                "admin",
                handlers.admin_command))
        application.add_handler(
            CommandHandler(
                "learn",
                handlers.learn_command))
        application.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                handlers.handle_text_message))
        application.add_handler(
            MessageHandler(
                filters.VOICE,
                handlers.handle_voice_message))
        application.add_handler(
            MessageHandler(
                filters.Document,
                handlers.handle_file_message))
        # Note: Ensure callback_query_handler is defined in AdvancedHandlers
        application.add_handler(
            CallbackQueryHandler(
                handlers.callback_query_handler))
        application.add_error_handler(handlers.error_handler)
        logger.debug("All handlers registered successfully")
    except Exception as e:
        logger.error(f"Error registering handlers: {e}")
        raise
