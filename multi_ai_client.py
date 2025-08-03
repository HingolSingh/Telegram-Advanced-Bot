"""
Multi-AI Client for integrating multiple AI providers.
Supports text generation, image generation, image analysis, audio transcription, web search, translation, code analysis, summarization, and math solving.
Providers: Gemini (default), Grok xAI, OpenAI, Anthropic, Stability AI, Cohere, Together.ai, DeepInfra, Hugging Face, OpenRouter, Groq Mixtral.
"""
import os
import asyncio
import logging
import aiohttp
import json
from typing import Dict, Any, Optional, List
from base64 import b64encode, b64decode
from PIL import Image
import io
import urllib.parse

logger = logging.getLogger(__name__)


class MultiAIClient:
    """Client for interacting with multiple AI providers with advanced capabilities."""

    def __init__(self):
        self.api_keys = {
            "gemini": os.getenv("GEMINI_API_KEY"),
            "grok": os.getenv("GROK_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "stability": os.getenv("STABILITY_API_KEY"),
            "cohere": os.getenv("COHERE_API_KEY"),
            "together": os.getenv("TOGETHER_API_KEY"),
            "deepinfra": os.getenv("DEEPINFRA_API_KEY"),
            "huggingface": os.getenv("HUGGINGFACE_API_KEY"),
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
            "groq": os.getenv("GROQ_API_KEY"),
        }

        self.base_urls = {
            "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
            "grok": "https://api.x.ai/v1",
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1",
            "stability": "https://api.stability.ai/v1",
            "cohere": "https://api.cohere.ai/v1",
            "together": "https://api.together.ai/v1",
            "deepinfra": "https://api.deepinfra.com/v1",
            "huggingface": "https://api-inference.huggingface.co/models",
            "openrouter": "https://openrouter.ai/api/v1",
            "groq": "https://api.groq.com/openai/v1",
        }

        self.models = {
            "gemini": {
                "name": "Gemini-Pro",
                "free": True,
                "capabilities": [
                    "text",
                    "summarization"]},
            "grok": {
                "name": "Grok xAI",
                "free": False,
                "capabilities": [
                    "text",
                    "image_analysis",
                    "web_search",
                    "code"]},
            "openai": {
                "name": "GPT-4o",
                        "free": False,
                        "capabilities": [
                            "text",
                            "image_analysis",
                            "audio",
                            "translation"]},
            "anthropic": {
                "name": "Claude-3-Opus",
                "free": False,
                "capabilities": [
                    "text",
                    "summarization"]},
            "stability": {
                "name": "Stable Diffusion XL",
                "free": False,
                "capabilities": ["image_generation"]},
            "cohere": {
                "name": "Cohere Command",
                "free": False,
                "capabilities": ["text"]},
            "together": {
                "name": "Mixtral-8x7B",
                "free": False,
                "capabilities": [
                    "text",
                    "image_generation"]},
            "deepinfra": {
                "name": "LLaMA-13B",
                "free": False,
                "capabilities": ["text"]},
            "huggingface": {
                "name": "Mistral-7B",
                "free": True,
                "capabilities": [
                    "text",
                    "image_generation"]},
            "openrouter": {
                "name": "OpenRouter Mix",
                "free": True,
                "capabilities": ["text"]},
            "groq": {
                "name": "Mixtral-8x7B",
                "free": False,
                "capabilities": [
                    "text",
                    "code"]},
        }

        self.default_model = "gemini"
        self.session_timeout = 30  # seconds

    async def _make_api_request(self,
                                url: str,
                                headers: Dict[str,
                                              str],
                                data: Optional[Dict] = None,
                                method: str = "POST",
                                files: Optional[Dict] = None) -> Dict[str,
                                                                      Any]:
        """Unified API request handler with error handling."""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.session_timeout)) as session:
            try:
                async with session.request(method, url, headers=headers, json=data, data=files) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 401:
                        logger.error(
                            f"Unauthorized: Invalid API key for {url}")
                        raise ValueError("Invalid API key")
                    elif response.status == 429:
                        logger.error(f"Rate limit exceeded for {url}")
                        raise ValueError("Rate limit exceeded")
                    else:
                        logger.error(f"API request failed: {response.status} - {await response.text()}")
                        raise Exception(
                            f"API request failed: {response.status}")
            except Exception as e:
                logger.error(f"API request error for {url}: {e}")
                raise

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Return available AI models and their capabilities."""
        return self.models

    async def generate_response(self,
                                user_id: str,
                                prompt: str,
                                model: str = "gemini",
                                conversation_history: Optional[List[Dict[str,
                                                                         Any]]] = None) -> str:
        """Generate text response from specified model with conversation context."""
        if model not in self.models:
            logger.error(f"Invalid model: {model}")
            return f"Error: Invalid model '{model}'. Use /ai to see available models."

        if not self.api_keys[model]:
            logger.error(f"No API key for {model}")
            return f"Error: API key not configured for {model}."

        headers = {
            "Authorization": f"Bearer {self.api_keys[model]}",
            "Content-Type": "application/json",
        }

        model_configs = {
            "gemini": {
                "url": f"{self.base_urls['gemini']}/gemini-pro:generateContent",
                "data": {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}] +
                    (conversation_history or []),
                    "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.7},
                },
            },
            "grok": {
                "url": f"{self.base_urls['grok']}/chat/completions",
                "data": {
                    "model": "grok-3",
                    "messages": (conversation_history or []) + [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                    "temperature": 0.7,
                },
            },
            "openai": {
                "url": f"{self.base_urls['openai']}/chat/completions",
                "data": {
                    "model": "gpt-4o",
                    "messages": (conversation_history or []) + [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                    "temperature": 0.7,
                },
            },
            "anthropic": {
                "url": f"{self.base_urls['anthropic']}/messages",
                "data": {
                    "model": "claude-3-opus-20240229",
                    "messages": (conversation_history or []) + [{"role": "user", "content": prompt}],
                    "max_tokens": 1024,
                    "temperature": 0.7,
                },
            },
            "cohere": {
                "url": f"{self.base_urls['cohere']}/generate",
                "data": {
                    "prompt": prompt,
                    "model": "command-xlarge-nightly",
                    "max_tokens": 500,
                    "temperature": 0.7,
                },
            },
            "together": {
                "url": f"{self.base_urls['together']}/completions",
                "data": {
                    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "prompt": f"[INST] {prompt} [/INST]",
                    "max_tokens": 500,
                    "temperature": 0.7,
                },
            },
            "deepinfra": {
                "url": f"{self.base_urls['deepinfra']}/completions",
                "data": {
                    "model": "meta-llama/Llama-2-13b-chat-hf",
                    "prompt": prompt,
                    "max_tokens": 500,
                    "temperature": 0.7,
                },
            },
            "huggingface": {
                "url": f"{self.base_urls['huggingface']}/mistralai/Mixtral-8x7B-Instruct-v0.1",
                "data": {
                    "inputs": f"[INST] {prompt} [/INST]",
                    "parameters": {"max_new_tokens": 500, "temperature": 0.7},
                },
            },
            "openrouter": {
                "url": f"{self.base_urls['openrouter']}/chat/completions",
                "data": {
                    "model": "mistralai/mixtral-8x7b-instruct",
                    "messages": (conversation_history or []) + [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                    "temperature": 0.7,
                },
            },
            "groq": {
                "url": f"{self.base_urls['groq']}/chat/completions",
                "data": {
                    "model": "mixtral-8x7b-32768",
                    "messages": (conversation_history or []) + [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                    "temperature": 0.7,
                },
            },
        }

        try:
            config = model_configs[model]
            response = await self._make_api_request(config["url"], headers, config["data"])

            if model == "gemini":
                return response["candidates"][0]["content"]["parts"][0]["text"]
            elif model in ["grok", "openai", "openrouter", "groq"]:
                return response["choices"][0]["message"]["content"]
            elif model == "anthropic":
                return response["content"][0]["text"]
            elif model in ["cohere", "together", "deepinfra"]:
                return response["generations"][0]["text"]
            elif model == "huggingface":
                return response[0]["generated_text"]
            else:
                return "Unexpected response format."
        except Exception as e:
            logger.error(
                f"Text generation error for {model} by user {user_id}: {e}")
            return f"Error: Failed to generate response with {model}. Try switching models with /ai."

    async def generate_image(
            self,
            user_id: str,
            prompt: str,
            model: str = "stability") -> Optional[bytes]:
        """Generate image from prompt using specified model."""
        if model not in ["stability", "huggingface", "together"]:
            logger.error(f"Invalid image generation model: {model}")
            return None

        if not self.api_keys[model]:
            logger.error(f"No API key for {model}")
            return None

        headers = {
            "Authorization": f"Bearer {self.api_keys[model]}",
            "Content-Type": "application/json",
        }

        model_configs = {
            "stability": {
                "url": f"{self.base_urls['stability']}/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                "data": {
                    "text_prompts": [{"text": prompt}],
                    "cfg_scale": 7,
                    "height": 1024,
                    "width": 1024,
                    "steps": 50,
                },
            },
            "huggingface": {
                "url": f"{self.base_urls['huggingface']}/stabilityai/stable-diffusion-2",
                "data": {
                    "inputs": prompt,
                    "parameters": {"num_inference_steps": 50, "guidance_scale": 7.5},
                },
            },
            "together": {
                "url": f"{self.base_urls['together']}/images/generations",
                "data": {
                    "model": "stabilityai/stable-diffusion-2",
                    "prompt": prompt,
                    "n": 1,
                    "size": "1024x1024",
                    "steps": 50,
                },
            },
        }

        try:
            config = model_configs[model]
            async with aiohttp.ClientSession() as session:
                async with session.post(config["url"], headers=headers, json=config["data"]) as response:
                    if response.status != 200:
                        logger.error(f"Image generation failed: {response.status} - {await response.text()}")
                        return None

                    if model == "stability":
                        data = await response.json()
                        image_data = b64decode(data["artifacts"][0]["base64"])
                    else:
                        image_data = await response.read()

                    image = Image.open(io.BytesIO(image_data))
                    output = io.BytesIO()
                    image.save(output, format="PNG")
                    return output.getvalue()
        except Exception as e:
            logger.error(
                f"Image generation error for {model} by user {user_id}: {e}")
            return None

    async def analyze_image(
            self,
            user_id: str,
            image_data: bytes,
            model: str = "grok") -> str:
        """Analyze image using vision-capable models."""
        if model not in ["grok", "openai"]:
            logger.error(f"Invalid image analysis model: {model}")
            return "Error: Image analysis not supported for this model."

        if not self.api_keys[model]:
            logger.error(f"No API key for {model}")
            return f"Error: API key not configured for {model}."

        headers = {
            "Authorization": f"Bearer {self.api_keys[model]}",
            "Content-Type": "application/json",
        }

        encoded_image = b64encode(image_data).decode("utf-8")
        model_configs = {
            "grok": {
                "url": f"{self.base_urls['grok']}/vision/analyze",
                "data": {
                    "image": f"data:image/jpeg;base64,{encoded_image}",
                    "prompt": "Describe this image in detail, including objects, colors, and context.",
                },
            },
            "openai": {
                "url": f"{self.base_urls['openai']}/chat/completions",
                "data": {
                    "model": "gpt-4-vision-preview",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image in detail."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                            ],
                        }
                    ],
                    "max_tokens": 500,
                },
            },
        }

        try:
            config = model_configs[model]
            response = await self._make_api_request(config["url"], headers, config["data"])
            return (
                response["description"] if model == "grok" else
                response["choices"][0]["message"]["content"]
            )
        except Exception as e:
            logger.error(
                f"Image analysis error for {model} by user {user_id}: {e}")
            return "Error: Failed to analyze image."

    async def transcribe_audio(
            self,
            user_id: str,
            audio_data: bytes,
            audio_format: str) -> Optional[str]:
        """Transcribe audio using OpenAI Whisper."""
        if not self.api_keys["openai"]:
            logger.error("No OpenAI API key")
            return "Error: OpenAI API key not configured."

        headers = {"Authorization": f"Bearer {self.api_keys['openai']}"}
        try:
            async with aiohttp.ClientSession() as session:
                form = aiohttp.FormData()
                form.add_field(
                    "file",
                    audio_data,
                    filename=f"audio.{audio_format}")
                form.add_field("model", "whisper-1")
                async with session.post(
                    f"{self.base_urls['openai']}/audio/transcriptions",
                    headers=headers,
                    data=form
                ) as response:
                    if response.status != 200:
                        logger.error(f"Audio transcription failed: {response.status} - {await response.text()}")
                        return None
                    data = await response.json()
                    return data.get("text", "No transcription available.")
        except Exception as e:
            logger.error(f"Audio transcription error for user {user_id}: {e}")
            return "Error: Failed to transcribe audio."

    async def web_search(
            self,
            user_id: str,
            query: str,
            model: str = "grok") -> str:
        """Perform web search using Grok xAI DeepSearch."""
        if model != "grok":
            logger.error(
                f"Web search only supported with Grok xAI, not {model}")
            return "Error: Web search is only supported with Grok xAI."

        if not self.api_keys[model]:
            logger.error(f"No API key for {model}")
            return f"Error: API key not configured for {model}."

        headers = {
            "Authorization": f"Bearer {self.api_keys[model]}",
            "Content-Type": "application/json",
        }
        data = {"query": urllib.parse.quote(query), "max_results": 5}

        try:
            response = await self._make_api_request(
                f"{self.base_urls['grok']}/search",
                headers,
                data
            )
            results = response.get("results", [])
            if not results:
                return "No search results found."

            return "\n".join(
                [f"• [{result['title']}]({result['url']}): {result['snippet']}" for result in results]
            )
        except Exception as e:
            logger.error(f"Web search error for user {user_id}: {e}")
            return "Error: Failed to perform web search."

    async def translate_text(
            self,
            user_id: str,
            text: str,
            target_lang: str) -> str:
        """Translate text to target language using OpenAI."""
        if not self.api_keys["openai"]:
            logger.error("No OpenAI API key")
            return "Error: OpenAI API key not configured."

        headers = {
            "Authorization": f"Bearer {self.api_keys['openai']}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": f"Translate '{text}' to {target_lang}."}
            ],
            "max_tokens": 500,
        }

        try:
            response = await self._make_api_request(
                f"{self.base_urls['openai']}/chat/completions",
                headers,
                data
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Translation error for user {user_id}: {e}")
            return "Error: Failed to translate text."

    async def analyze_code(
            self,
            user_id: str,
            code: str,
            language: str) -> str:
        """Analyze and improve code using Grok xAI."""
        if not self.api_keys["grok"]:
            logger.error("No Grok API key")
            return "Error: Grok API key not configured."

        headers = {
            "Authorization": f"Bearer {self.api_keys['grok']}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "grok-3",
            "messages": [
                {
                    "role": "user",
                    "content": f"Analyze and improve this {language} code. Provide suggestions and explanations:\n``` {code} ```"
                }
            ],
            "max_tokens": 2048,
        }

        try:
            response = await self._make_api_request(
                f"{self.base_urls['grok']}/chat/completions",
                headers,
                data
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Code analysis error for user {user_id}: {e}")
            return "Error: Failed to analyze code."

    async def summarize_text(
            self,
            user_id: str,
            text: str,
            model: str = "gemini") -> str:
        """Summarize text using specified model."""
        if model not in self.models:
            logger.error(f"Invalid model: {model}")
            return f"Error: Invalid model '{model}'."

        if not self.api_keys[model]:
            logger.error(f"No API key for {model}")
            return f"Error: API key not configured for {model}."

        headers = {
            "Authorization": f"Bearer {self.api_keys[model]}",
            "Content-Type": "application/json",
        }
        model_configs = {
            "gemini": {
                "url": f"{self.base_urls['gemini']}/gemini-pro:generateContent",
                "data": {
                    "contents": [{"role": "user", "parts": [{"text": f"Summarize this text in 100 words or less:\n{text}"}]}],
                },
            },
            "grok": {
                "url": f"{self.base_urls['grok']}/chat/completions",
                "data": {
                    "model": "grok-3",
                    "messages": [{"role": "user", "content": f"Summarize this text in 100 words or less:\n{text}"}],
                    "max_tokens": 200,
                },
            },
        }

        try:
            config = model_configs.get(model, model_configs["gemini"])
            response = await self._make_api_request(config["url"], headers, config["data"])
            return (
                response["candidates"][0]["content"]["parts"][0]["text"] if model == "gemini" else
                response["choices"][0]["message"]["content"]
            )
        except Exception as e:
            logger.error(
                f"Summarization error for {model} by user {user_id}: {e}")
            return "Error: Failed to summarize text."

    async def solve_math(
            self,
            user_id: str,
            problem: str,
            model: str = "grok") -> str:
        """Solve math problems step-by-step using specified model."""
        if model not in self.models:
            logger.error(f"Invalid model: {model}")
            return f"Error: Invalid model '{model}'."

        if not self.api_keys[model]:
            logger.error(f"No API key for {model}")
            return f"Error: API key not configured for {model}."

        headers = {
            "Authorization": f"Bearer {self.api_keys[model]}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "grok-3" if model == "grok" else "gpt-4o",
            "messages": [
                {"role": "user", "content": f"Solve this math problem step-by-step:\n{problem}"}
            ],
            "max_tokens": 2048,
        }

        try:
            response = await self._make_api_request(
                f"{self.base_urls[model]}/chat/completions",
                headers,
                data
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(
                f"Math solving error for {model} by user {user_id}: {e}")
            return "Error: Failed to solve math problem."
