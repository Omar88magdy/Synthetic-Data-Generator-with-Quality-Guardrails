"""
Model providers interface for generating synthetic reviews.
"""

import os
import time
import json
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass

from .models import ModelMetadata, Persona, Review


@dataclass
class GenerationRequest:
    """Request for review generation."""
    persona: Persona
    rating: int
    domain: str
    domain_keywords: List[str]
    target_length: str  # "short", "medium", "long"
    sentiment: str
    tool_name: Optional[str] = None


class ModelProvider(ABC):
    """Abstract base class for model providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def generate_review(self, request: GenerationRequest) -> tuple[str, ModelMetadata]:
        """Generate a review based on the request."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        pass
    
    def _create_prompt(self, request: GenerationRequest) -> str:
        """Create a prompt for review generation."""
        length_guidance = {
            "short": "Write a concise review (50-120 words)",
            "medium": "Write a detailed review (120-250 words)", 
            "long": "Write a comprehensive review (250-400 words)"
        }
        
        sentiment_guidance = {
            "positive": "Focus on strengths and benefits",
            "negative": "Focus on limitations and issues",
            "neutral": "Provide a balanced perspective"
        }
        
        tool_name = request.tool_name or "easygenerator"
        
        prompt = f"""You are a {request.persona.role} with {request.persona.experience.value} experience. 
Write a {request.rating}-star review for {tool_name}, a {request.domain.replace('_', ' ')} solution.

Your perspective should be {request.persona.tone.value} and incorporate these characteristics:
{chr(10).join('- ' + char for char in request.persona.characteristics)}

{length_guidance[request.target_length]} that {sentiment_guidance[request.sentiment]}.

Include relevant technical details from this domain: {', '.join(request.domain_keywords[:8])}

Write naturally as if sharing your genuine experience. Be specific about features, use cases, and impact.
Do not mention that this is a synthetic review or that you are an AI and use diverse vocabulary.

Review:"""
        
        return prompt
    
    def is_available(self) -> bool:
        """Check if the provider is available."""
        return self.config.get("enabled", False)


class OpenAIProvider(ModelProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = os.getenv(config.get("api_key_env", "OPENAI_API_KEY"))
        self.model = config.get("model", "gpt-4o-mini")
        self.base_url = "https://api.openai.com/v1"
        
    @property
    def provider_name(self) -> str:
        return "openai"
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return super().is_available() and self.api_key is not None
    
    def generate_review(self, request: GenerationRequest) -> tuple[str, ModelMetadata]:
        """Generate review using OpenAI API."""
        start_time = time.time()
        
        try:
            prompt = self._create_prompt(request)
            # print(f'-------\nprompt is {prompt}\n-------')
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": self.config.get("temperature", 0.8),
                "max_tokens": self.config.get("max_tokens", 300)
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            result = response.json()
            
            generation_time_ms = int((time.time() - start_time) * 1000)
            
            review_text = result["choices"][0]["message"]["content"].strip()
            
            # Estimate cost (rough approximation)
            prompt_tokens = result.get("usage", {}).get("prompt_tokens", 0)
            completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
            
            # Cost estimation for gpt-4o-mini (as of 2026)
            cost_per_1k_prompt = 0.00015
            cost_per_1k_completion = 0.0006
            estimated_cost = (prompt_tokens * cost_per_1k_prompt / 1000 + 
                            completion_tokens * cost_per_1k_completion / 1000)
            
            metadata = ModelMetadata(
                provider=self.provider_name,
                model_name=self.model,
                temperature=self.config.get("temperature", 0.8),
                max_tokens=self.config.get("max_tokens", 300),
                generation_time_ms=generation_time_ms,
                api_cost_estimate=estimated_cost
            )
            
            self.logger.info(f"Generated review using OpenAI in {generation_time_ms}ms")
            
            return review_text, metadata
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"OpenAI API request failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"OpenAI generation failed: {e}")
            raise

class DeepSeekProvider(ModelProvider):
    """DeepSeek API provider using the OpenAI SDK format."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = os.getenv(config.get("api_key_env", "DEEPSEEK_API_KEY"))
        self.model = config.get("model", "deepseek-chat")
        self.base_url = "https://api.deepseek.com"
        
    @property
    def provider_name(self) -> str:
        return "deepseek"
    
    def is_available(self) -> bool:
        """Check if DeepSeek is available."""
        return super().is_available() and self.api_key is not None
    
    def generate_review(self, request: GenerationRequest) -> tuple[str, ModelMetadata]:
        """Generate review using DeepSeek API via OpenAI SDK."""
        start_time = time.time()
        
        try:
            # Import OpenAI here to avoid requiring it globally
            from openai import OpenAI
            
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            prompt = self._create_prompt(request)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.config.get("temperature", 0.8),
                max_tokens=self.config.get("max_tokens", 500),
                stream=False
            )
            
            generation_time_ms = int((time.time() - start_time) * 1000)
            
            review_text = response.choices[0].message.content.strip()
            
            # Estimate cost (rough approximation for DeepSeek)
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            
            # DeepSeek pricing (approximate)
            cost_per_1k_prompt = 0.00014  # Competitive pricing
            cost_per_1k_completion = 0.00028
            estimated_cost = (prompt_tokens * cost_per_1k_prompt / 1000 + 
                            completion_tokens * cost_per_1k_completion / 1000)
            
            metadata = ModelMetadata(
                provider=self.provider_name,
                model_name=self.model,
                temperature=self.config.get("temperature", 0.8),
                max_tokens=self.config.get("max_tokens", 500),
                generation_time_ms=generation_time_ms,
                api_cost_estimate=estimated_cost
            )
            
            self.logger.info(f"Generated review using DeepSeek in {generation_time_ms}ms")
            
            return review_text, metadata
            
        except Exception as e:
            self.logger.error(f"DeepSeek generation failed: {e}")
            raise

class ModelProviderFactory:
    """Factory for creating model providers."""
    
    PROVIDERS = {
        "openai": OpenAIProvider,
        "deepseek": DeepSeekProvider
    }
    
    @classmethod
    def create_provider(cls, provider_name: str, config: Dict[str, Any]) -> Optional[ModelProvider]:
        """Create a model provider instance."""
        if provider_name not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        provider_class = cls.PROVIDERS[provider_name]
        provider = provider_class(config)
        
        if not provider.is_available():
            logging.warning(f"Provider {provider_name} is not available")
            return None
        
        return provider
    
    @classmethod
    def create_all_providers(cls, config: Dict[str, Dict[str, Any]]) -> Dict[str, ModelProvider]:
        """Create all available providers from config."""
        providers = {}
        
        for provider_name, provider_config in config.items():
            try:
                provider = cls.create_provider(provider_name, provider_config)
                if provider:
                    providers[provider_name] = provider
            except Exception as e:
                logging.error(f"Failed to create provider {provider_name}: {e}")
        
        return providers
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of all available provider names."""
        return list(cls.PROVIDERS.keys())