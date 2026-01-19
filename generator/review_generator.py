"""
Main review generator that orchestrates the synthetic review creation process.
"""

import random
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import yaml

from .models import Review, ReviewDataset, GenerationStats, SentimentType
from .model_providers import ModelProviderFactory, GenerationRequest
from .personas import PersonaManager


class ReviewGenerator:
    """Main class for generating synthetic reviews."""
    
    def __init__(self, config_path: str):
        """Initialize the review generator with configuration."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.persona_manager = PersonaManager(self.config)
        self.model_providers = ModelProviderFactory.create_all_providers(
            self.config.get("models", {})
        )
        
        if not self.model_providers:
            raise RuntimeError("No model providers are available")
        
        self.logger.info(f"Initialized with {len(self.model_providers)} model providers")
        
        # Generation settings
        self.target_reviews = self.config["generation"]["target_reviews"]
        self.min_reviews = self.config["generation"]["min_reviews"]
        self.batch_size = self.config["generation"]["batch_size"]
        
        # Domain settings
        self.domain = self.config["domain"]
        self.domain_keywords = self.config["domain_keywords"]
        
        # Statistics tracking
        self.stats = GenerationStats()
        
    def generate_dataset(self, progress_callback: Optional[callable] = None) -> ReviewDataset:
        """Generate a complete dataset of synthetic reviews."""
        self.logger.info(f"Starting generation of {self.target_reviews} reviews")
        
        dataset = ReviewDataset()
        generated_count = 0
        
        # Calculate provider distribution
        provider_weights = {name: config.get("weight", 1.0) 
                          for name, config in self.config["models"].items() 
                          if config.get("enabled", False)}
        
        total_weight = sum(provider_weights.values())
        provider_distribution = {name: weight / total_weight 
                               for name, weight in provider_weights.items()}
        
        while generated_count < self.target_reviews:
            batch_size = min(self.batch_size, self.target_reviews - generated_count)
            batch_reviews = self._generate_batch(batch_size, provider_distribution)
            
            for review in batch_reviews:
                dataset.add_review(review)
                generated_count += 1
                
                if progress_callback:
                    progress_callback(generated_count, self.target_reviews)
                
                self.logger.debug(f"Generated review {generated_count}/{self.target_reviews}")
        
        # Update dataset metadata
        dataset.generation_stats = self.stats
        dataset.metadata = {
            "generation_timestamp": datetime.now().isoformat(),
            "config_domain": self.domain,
            "target_count": self.target_reviews,
            "actual_count": generated_count,
            "model_providers": list(self.model_providers.keys())
        }
        
        self.logger.info(f"Completed generation of {generated_count} reviews")
        return dataset
    
    def _generate_batch(self, batch_size: int, provider_distribution: Dict[str, float]) -> List[Review]:
        """Generate a batch of reviews."""
        reviews = []
        
        for _ in range(batch_size):
            try:
                review = self._generate_single_review(provider_distribution)
                if review:
                    reviews.append(review)
            except Exception as e:
                self.logger.error(f"Failed to generate review: {e}")
                self.stats.add_generation_attempt(
                    success=False, 
                    generation_time_ms=0, 
                    model_provider="unknown"
                )
        
        return reviews
    
    def _generate_single_review(self, provider_distribution: Dict[str, float]) -> Optional[Review]:
        """Generate a single review."""
        # Select model provider
        provider_name = random.choices(
            list(provider_distribution.keys()),
            weights=list(provider_distribution.values()),
            k=1
        )[0]
        
        provider = self.model_providers[provider_name]
        
        # Generate review parameters
        persona = self.persona_manager.get_random_persona()
        rating = self._select_rating()
        target_length = self._select_length()
        sentiment = self._determine_sentiment(rating)
        
        # Create generation request
        request = GenerationRequest(
            persona=persona,
            rating=rating,
            domain=self.domain,
            domain_keywords=self.domain_keywords,
            target_length=target_length,
            sentiment=sentiment,
            tool_name=None
        )
        
        try:
            # Generate review
            review_text, model_metadata = provider.generate_review(request)
            
            # Create review object
            review = Review(
                id="",  # Will be auto-generated
                rating=rating,
                review_text=review_text,
                timestamp=datetime.now(),
                persona=persona,
                model_metadata=model_metadata,
                sentiment=SentimentType(sentiment),
                is_synthetic=True
            )
            
            # Record successful generation
            self.stats.add_generation_attempt(
                success=True,
                generation_time_ms=model_metadata.generation_time_ms,
                model_provider=provider_name
            )
            
            return review
            
        except Exception as e:
            self.logger.error(f"Generation failed with {provider_name}: {e}")
            self.stats.add_generation_attempt(
                success=False,
                generation_time_ms=0,
                model_provider=provider_name
            )
            return None
    
    def _select_rating(self) -> int:
        """Select a rating based on configured distribution."""
        rating_dist = self.config["rating_distribution"]
        ratings = list(rating_dist.keys())
        weights = list(rating_dist.values())
        
        return random.choices(ratings, weights=weights, k=1)[0]
    
    def _select_length(self) -> str:
        """Select review length based on configured distribution."""
        length_dist = self.config["review_settings"]["length_distribution"]
        lengths = list(length_dist.keys())
        weights = list(length_dist.values())
        
        return random.choices(lengths, weights=weights, k=1)[0]
    
    def _determine_sentiment(self, rating: int) -> str:
        """Determine sentiment based on rating with some randomness."""
        # Base sentiment mapping
        if rating >= 4:
            base_sentiment = "positive"
        elif rating <= 2:
            base_sentiment = "negative"
        else:
            base_sentiment = "neutral"
        
        # Add some randomness based on sentiment mix configuration
        sentiment_mix = self.config["review_settings"]["sentiment_mix"]
        
        # Small chance to deviate from base sentiment for realism
        if random.random() < 0.15:  # 15% chance to deviate
            sentiments = list(sentiment_mix.keys())
            weights = list(sentiment_mix.values())
            return random.choices(sentiments, weights=weights, k=1)[0]
        
        return base_sentiment
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get current generation statistics."""
        return self.stats.to_dict()
    
    def estimate_generation_time(self, review_count: int) -> int:
        """Estimate total generation time in seconds."""
        if self.stats.average_generation_time_ms == 0:
            # Use default estimate if no stats available
            avg_time_ms = 3000  # 3 seconds per review
        else:
            avg_time_ms = self.stats.average_generation_time_ms
        
        total_time_ms = review_count * avg_time_ms
        return int(total_time_ms / 1000)
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the configuration for generation."""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check model providers
        if not self.model_providers:
            validation_results["errors"].append("No model providers are available")
            validation_results["valid"] = False
        
        # Check personas
        if not self.persona_manager.get_all_personas():
            validation_results["errors"].append("No personas are configured")
            validation_results["valid"] = False
        
        # Check rating distribution
        rating_sum = sum(self.config["rating_distribution"].values())
        if abs(rating_sum - 1.0) > 0.01:
            validation_results["warnings"].append(
                f"Rating distribution sums to {rating_sum}, should be 1.0"
            )
        
        # Check length distribution
        length_sum = sum(self.config["review_settings"]["length_distribution"].values())
        if abs(length_sum - 1.0) > 0.01:
            validation_results["warnings"].append(
                f"Length distribution sums to {length_sum}, should be 1.0"
            )
        
        # Check sentiment distribution
        sentiment_sum = sum(self.config["review_settings"]["sentiment_mix"].values())
        if abs(sentiment_sum - 1.0) > 0.01:
            validation_results["warnings"].append(
                f"Sentiment distribution sums to {sentiment_sum}, should be 1.0"
            )
        
        # Check domain keywords
        if len(self.domain_keywords) < 5:
            validation_results["warnings"].append(
                "Consider adding more domain keywords for better variety"
            )
        
        return validation_results