"""
LLM Client Utility

Provides a wrapper around local Ollama API to interact with LLMs like Mistral.
Supports both full-response and streaming modes.

Example:
    client = MistralClient(model="mistral")
    output = client.generate("Summarize this dataset...", stream=False)
"""

#Import libraries
import requests
from typing import Optional
import os
from dotenv import load_dotenv

#Import util libraries
from src.utils.logging import get_logger
from src.utils.exceptions import LLMError

load_dotenv()
logger = get_logger(__name__)

class MistralClient:
    """
    Client to communicate with local Mistral (or other Ollama models).
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral"):
        """
        Args:
            base_url (str): Base URL of the Ollama server.
            model (str): Name of the LLM to use (must be pulled in Ollama).
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        logger.info(f"MistralClient initialized with model '{self.model}' at {self.base_url}")

    def generate(self, prompt: str, stream: bool = False, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        """
        Generate a response from the Mistral model.

        Args:
            prompt (str): Prompt to send to the model.
            stream (bool): Whether to stream the response.
            temperature (float): Sampling temperature (higher = more random).
            max_tokens (Optional[int]): Max tokens to generate (None = model default).

        Returns:
            str: Generated response text.

        Raises:
            LLMError: If the request fails.
        """
        logger.debug(f"Sending prompt to Mistral model: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature
                }
            }
            if max_tokens is not None:
                payload["options"]["num_predict"] = max_tokens

            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()

            if stream:
                output = ""
                for line in resp.iter_lines():
                    if line:
                        chunk = line.decode("utf-8")
                        output += chunk
                        logger.debug(f"Stream chunk: {chunk}")
                return output
            else:
                data = resp.json()
                return data.get("response", "")

        except requests.RequestException as e:
            logger.error(f"Request to Mistral model failed: {e}")
            raise LLMError(f"Failed to connect to LLM at {self.base_url}") from e
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise LLMError(f"LLM generation failed: {e}")

    def chat(self, messages: list, temperature: float = 0.7) -> str:
        """
        Chat-style interface for LLM.

        Args:
            messages (list): List of dict messages: [{"role": "user", "content": "Hello"}]
            temperature (float): Sampling temperature.

        Returns:
            str: Model's reply.
        """
        try:
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": messages,
                "options": {
                    "temperature": temperature
                }
            }
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            raise LLMError(f"LLM chat request failed: {e}")

class GeminiClient:
    def __init__(self, model="gemini-pro", api_key=None):
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        logger.info(f"GeminiClient initialized with model '{self.model}'")

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
        try:
            payload = {
                "contents": [
                    {
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens
                }
            }
            params = {"key": self.api_key}
            resp = requests.post(self.base_url, params=params, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise LLMError(f"Gemini generation failed: {e}")