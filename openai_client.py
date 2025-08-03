"""
OpenAI API client for generating conversational responses.
"""
import logging
import asyncio
from openai import OpenAI
from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

# Initialize OpenAI client
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable is required")
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)


async def get_ai_response(conversation_history: list) -> str:
    """
    Get AI response from OpenAI API using conversation history.

    Args:
        conversation_history: List of message dictionaries with 'role' and 'content'

    Returns:
        AI response as string

    Raises:
        Exception: If API call fails or client is not initialized
    """
    if not client:
        raise Exception("OpenAI client not initialized - check OPENAI_API_KEY")

    try:
        # Run the synchronous OpenAI call in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI modelis "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=conversation_history,
                max_tokens=1500,
                temperature=0.7,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
        )

        ai_response = response.choices[0].message.content

        if not ai_response:
            raise Exception("Empty response from OpenAI API")

        logger.debug(
            f"Generated AI response with {len(ai_response)} characters")
        return ai_response.strip()

    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")

        # Check for specific error types and provide more helpful messages
        error_message = str(e).lower()

        if "rate limit" in error_message:
            raise Exception(
                "API rate limit exceeded. Please try again in a few minutes.")
        elif "insufficient" in error_message and "quota" in error_message:
            raise Exception(
                "API quota exceeded. Please contact the bot administrator.")
        elif "invalid" in error_message and "key" in error_message:
            raise Exception(
                "API authentication failed. Please contact the bot administrator.")
        elif "connection" in error_message or "timeout" in error_message:
            raise Exception(
                "Connection to AI service failed. Please try again.")
        else:
            raise Exception(f"AI service error: {str(e)}")
