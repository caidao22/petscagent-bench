"""LLM client for evaluation purposes."""

import os
import json
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel
import openai


class LLMClient:
    """Client for making LLM API calls."""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3):
        """Initialize LLM client.
        
        Args:
            model: OpenAI model name
            temperature: Sampling temperature (0-1)
        """
        self.model = model
        self.temperature = temperature
        self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    async def structured_completion(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        system_prompt: Optional[str] = None
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
        
        # Use JSON mode
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        
        # Parse JSON response
        content = response.choices[0].message.content
        data = json.loads(content)
        
        # Validate with pydantic model
        return response_model(**data)
    
    async def completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
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
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        
        return response.choices[0].message.content