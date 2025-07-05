import abc
import logging
import openai
from abc import abstractmethod
from rag.config import OPENAI_API_KEY
from typing import Optional

logger = logging.getLogger(__name__)

class LLM(abc.ABC):    
    def __init__(self):
        pass

    @abstractmethod
    def generate_response(self, prompt: str, model_name: str) -> str:
        ...
    
    def _log_prompt_and_response(self, prompt: str, response: str):
        logger.debug(f"LLM Prompt: {prompt}")
        logger.debug(f"LLM Response: {response}")

    @abstractmethod
    def _validate_config(self) -> None:
        ...

    @abstractmethod
    def _validate_connectivity(self) -> None:
        ...

class OpenAI_LLM(LLM):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or OPENAI_API_KEY
        self._validate_config()
        self.client = openai.OpenAI(api_key=self.api_key)
        self._validate_connectivity()

    def _validate_config(self) -> None:
        if not self.api_key:
            raise ValueError("API key is required but not provided")
        if not isinstance(self.api_key, str):
            raise ValueError("API key must be a string")
        if not self.api_key.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        logger.info(f"Successfully validated OpenAI configuration")
    
    def _validate_connectivity(self) -> None:
        try:
            self.client.models.list()
        except Exception as e:
            self.handle_openai_error(e)

    def generate_response(self, prompt: str, model_name: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            if not response.choices or response.choices[0].message.content is None:
                return "No response from OpenAI"
            self._log_prompt_and_response(prompt, response.choices[0].message.content)
            return response.choices[0].message.content
        except Exception as e:
            self.handle_openai_error(e)

    def handle_openai_error(self, error: Exception) -> None:
        if isinstance(error, openai.AuthenticationError):
            logger.error("OpenAI authentication error - check API key")
        elif isinstance(error, openai.RateLimitError):
            logger.error("Rate limit exceeded")
        elif isinstance(error, openai.APIError):
            logger.error(f"OpenAI API error: {error}")
        else:
            logger.error(f"Unexpected error: {error}")
        raise error