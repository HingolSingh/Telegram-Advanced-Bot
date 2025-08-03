"""
Gemini API client for generating conversational responses.
"""
import logging
import asyncio
from google import genai
from google.genai import types
from config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

# Initialize Gemini client
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable is required")
    client = None
else:
    client = genai.Client(api_key=GEMINI_API_KEY)


async def get_ai_response(conversation_history: list) -> str:
    """
    Get AI response from Gemini API using conversation history.

    Args:
        conversation_history: List of message dictionaries with 'role' and 'content'

    Returns:
        AI response as string

    Raises:
        Exception: If API call fails or client is not initialized
    """
    if not client:
        raise Exception("Gemini client not initialized - check GEMINI_API_KEY")

    try:
        # Convert OpenAI format to Gemini format
        gemini_messages = []
        system_prompt = ""

        for msg in conversation_history:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                gemini_messages.append(
                    types.Content(
                        role="user", parts=[
                            types.Part(
                                text=msg["content"])]))
            elif msg["role"] == "assistant":
                gemini_messages.append(
                    types.Content(
                        role="model", parts=[
                            types.Part(
                                text=msg["content"])]))

        # Run the synchronous Gemini call in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model="gemini-2.5-flash",
                contents=gemini_messages,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                    max_output_tokens=1500,
                    top_p=0.95,
                    top_k=40
                )
            )
        )

        ai_response = response.text

        if not ai_response:
            raise Exception("Empty response from Gemini API")

        logger.debug(
            f"Generated AI response with {len(ai_response)} characters")
        return ai_response.strip()

    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")

        # Check for specific error types and provide more helpful messages
        error_message = str(e).lower()

        if "rate limit" in error_message or "quota" in error_message:
            raise Exception(
                "API rate limit exceeded. Please try again in a few minutes.")
        elif "invalid" in error_message and ("key" in error_message or "api" in error_message):
            raise Exception(
                "API authentication failed. Please check your Gemini API key.")
        elif "connection" in error_message or "timeout" in error_message:
            raise Exception(
                "Connection to AI service failed. Please try again.")
        else:
            raise Exception(f"AI service error: {str(e)}")
