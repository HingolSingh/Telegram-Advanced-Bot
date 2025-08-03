# Telegram AI Conversational Bot

A personal Telegram bot powered by Google's Gemini 2.5 Flash for intelligent conversational interactions - completely free to use!

## Features

- 🤖 **Intelligent Conversations**: Powered by Google's Gemini 2.5 Flash model (free tier)
- 💬 **Context Awareness**: Maintains conversation context within sessions
- ⚡ **Rate Limiting**: Prevents API abuse with configurable limits
- 🛡️ **Error Handling**: Graceful error handling with helpful fallback messages
- 📝 **Logging**: Comprehensive logging for debugging and monitoring
- 🔧 **Easy Configuration**: Environment variable-based configuration

## Commands

- `/start` - Initialize/reset conversation and show welcome message
- `/help` - Display help information and bot features
- Send any text message to chat with the AI

## Setup Instructions

### 1. Prerequisites

- Python 3.9 or higher
- Telegram account
- Google account (for free Gemini API)

### 2. Create a Telegram Bot

1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Use `/newbot` command and follow the instructions
3. Save the bot token provided by BotFather

### 3. Get Free Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Save the API key securely (completely free - no billing required!)

### 4. Environment Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` file with your credentials:
   ```
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
   GEMINI_API_KEY=your_gemini_api_key_here
   