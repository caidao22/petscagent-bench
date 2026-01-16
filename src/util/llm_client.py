import os
import json
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel
from litellm import acompletion
import dotenv

dotenv.load_dotenv()


class LLMClient:
    """Client for making LLM API calls."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3):
        """Initialize LLM client.

        Args:
            model: Model name (supports OpenAI, Anthropic, etc. via LiteLLM)
            temperature: Sampling temperature (0-1)
        """
        self.model = model
        self.temperature = temperature

    async def structured_completion(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        system_prompt: Optional[str] = None,
    ) -> BaseModel:
        """Get structured response from LLM.

        Args:
            prompt: User prompt
            response_model: Pydantic model for response structure
            system_prompt: Optional system prompt

        Returns:
            Parsed response as response_model instance
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            # Use JSON mode
            response = await acompletion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            # Parse JSON response
            content = response.choices[0].message.content
            data = json.loads(content)

            # Handle case where LLM returns array instead of object
            if isinstance(data, list) and data:
                # Use first element if it's an array of objects
                data = data[0]

            # Validate with pydantic model
            return response_model(**data)
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse JSON from LLM response")
            print(f"Raw content: {content[:500]}")
            raise
        except Exception as e:
            print(f"ERROR in structured_completion: {type(e).__name__}: {str(e)}")
            if 'content' in locals():
                print(f"Raw LLM response: {content[:500]}")
            if 'data' in locals():
                print(f"Parsed data type: {type(data)}, value: {str(data)[:200]}")
            raise

    async def completion(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Get text completion from LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Response text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = await acompletion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        return response.choices[0].message.content
