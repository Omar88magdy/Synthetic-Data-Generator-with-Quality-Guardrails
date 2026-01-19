"""
Simplified bias detection focused on sentiment skew.
"""

import numpy as np
from typing import Dict, List, Any
from collections import Counter
import logging

from generator.models import Review, SentimentType


class BiasDetector:
    """Detects sentiment bias in synthetic review generation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Expected sentiment distribution from config
        self.expected_sentiment_mix = config.get("review_settings", {}).get("sentiment_mix", {
            "positive": 0.75,   # Most reviews should be positive
            "neutral": 0.20,    # Some neutral/constructive reviews
            "negative": 0.05    # Few truly negative reviews
        })
    
    def analyze_sentiment_bias(self, reviews: List[Review]) -> Dict[str, Any]:
        """Analyze sentiment distribution for bias detection."""
        if not reviews:
            return {"status": "no_data", "overall_bias": 0.0}
        
        # Count sentiments
        sentiment_counts = Counter()
        for review in reviews:
            sentiment = review.sentiment.value if hasattr(review.sentiment, 'value') else str(review.sentiment)
            sentiment_counts[sentiment] += 1
        
        total_reviews = len(reviews)
        
        # Calculate actual distribution
        actual_distribution = {}
        for sentiment in ["positive", "neutral", "negative"]:
            actual_distribution[sentiment] = sentiment_counts.get(sentiment, 0) / total_reviews
        
        # Calculate bias score (deviation from expected distribution)
        bias_score = 0.0
        for sentiment, expected_ratio in self.expected_sentiment_mix.items():
            actual_ratio = actual_distribution.get(sentiment, 0.0)
            bias_score += abs(actual_ratio - expected_ratio)
        
        # Normalize bias score (0 = no bias, 1 = maximum bias)
        bias_score = bias_score / 2.0  # Max possible deviation is 2.0
        
        return {
            "status": "success",
            "overall_bias": bias_score,
            "expected_distribution": self.expected_sentiment_mix,
            "actual_distribution": actual_distribution,
            "total_reviews": total_reviews,
            "bias_level": self._classify_bias_level(bias_score)
        }
    
    def detect_bias(self, reviews: List[str]) -> Dict[str, Any]:
        """Main method for bias detection - converts strings to Review objects if needed."""
        if not reviews:
            return {"overall_bias": 0.0}
        
        # If we get strings instead of Review objects, create simple Review objects
        if isinstance(reviews[0], str):
            # Simple sentiment detection based on rating patterns in text
            review_objects = []
            for i, text in enumerate(reviews):
                sentiment = self._simple_sentiment_detection(text)
                review = Review(
                    id=str(i),
                    rating=4,  # Default rating
                    review_text=text,
                    sentiment=SentimentType(sentiment),
                    is_synthetic=True
                )
                review_objects.append(review)
            reviews = review_objects
        
        return self.analyze_sentiment_bias(reviews)
    
    def _simple_sentiment_detection(self, text: str) -> str:
        """Simple sentiment detection based on keywords."""
        text_lower = text.lower()
        
        # Positive indicators
        positive_words = ["great", "excellent", "amazing", "love", "perfect", "fantastic", "wonderful", "awesome"]
        negative_words = ["bad", "terrible", "awful", "hate", "horrible", "disappointing", "frustrating", "poor"]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _classify_bias_level(self, bias_score: float) -> str:
        """Classify bias level based on score."""
        if bias_score < 0.1:
            return "low"
        elif bias_score < 0.3:
            return "moderate"
        else:
            return "high"


def create_bias_detector(config: Dict[str, Any]) -> BiasDetector:
    """Factory function to create a BiasDetector."""
    return BiasDetector(config)