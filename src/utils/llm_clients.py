"""
LLM Client Utility

Provides a wrapper around local Ollama API to interact with LLMs like Mistral.
Supports both full-response and streaming modes.

Example:
    client = MistralClient(model="mistral")
    output = client.generate("Summarize this dataset...", stream=False)
"""

import requests
from typing import Optional
from src.utils.logging import get_logger
from src.utils.exceptions import LLMError

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
