"""
Simplified realism analysis focused on domain keyword relevance.
"""

import re
from typing import Dict, List, Set, Any
import logging
from datetime import datetime

from generator.models import Review


class RealismAnalyzer:
    """Analyzes domain relevance using keyword matching."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.domain_keywords = set(config.get("domain_keywords", []))
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Convert keywords to lowercase for case-insensitive matching
        self.domain_keywords_lower = {keyword.lower() for keyword in self.domain_keywords}
    
    def analyze_domain_relevance(self, review: Review) -> Dict[str, Any]:
        """Analyze how well a review matches the target domain using keyword matching."""
        text = review.review_text.lower()
        
        # Find which domain keywords are present
        found_keywords = []
        for keyword in self.domain_keywords_lower:
            if keyword in text:
                found_keywords.append(keyword)
        
        # Calculate relevance score
        if not self.domain_keywords:
            relevance_score = 1.0  # If no keywords defined, assume relevant
        else:
            relevance_score = len(found_keywords) / len(self.domain_keywords)
        
        return {
            "relevance_score": relevance_score,
            "found_keywords": found_keywords,
            "total_keywords": len(self.domain_keywords),
            "keywords_found": len(found_keywords)
        }
    
    def analyze_realism(self, reviews: List[str], real_reviews: List[str] = None) -> Dict[str, Any]:
        """Main method for realism analysis."""
        if not reviews:
            return {"overall_realism": 1.0}
        
        # Convert strings to Review objects if needed
        if isinstance(reviews[0], str):
            review_objects = []
            for i, text in enumerate(reviews):
                review = Review(
                    id=str(i),
                    rating=4,  # Default rating
                    review_text=text,
                    timestamp=datetime.now(),  # Default timestamp
                    word_count=len(text.split()),
                    is_synthetic=True
                )
                review_objects.append(review)
            reviews = review_objects
        
        # Analyze domain relevance for all reviews
        relevance_scores = []
        total_keywords_found = 0
        
        for review in reviews:
            analysis = self.analyze_domain_relevance(review)
            relevance_scores.append(analysis["relevance_score"])
            total_keywords_found += analysis["keywords_found"]
        
        # Calculate overall realism score
        overall_realism = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        return {
            "overall_realism": overall_realism,
            "avg_relevance_score": overall_realism,
            "total_reviews": len(reviews),
            "avg_keywords_per_review": total_keywords_found / len(reviews) if reviews else 0,
            "realism_level": self._classify_realism_level(overall_realism)
        }
    
    def calculate_overall_realism_score(self, review: Review) -> Dict[str, Any]:
        """Calculate overall realism score for a single review."""
        domain_analysis = self.analyze_domain_relevance(review)
        
        return {
            "overall_score": domain_analysis["relevance_score"],
            "domain_relevance": domain_analysis["relevance_score"],
            "found_keywords": domain_analysis["found_keywords"],
            "realism_level": self._classify_realism_level(domain_analysis["relevance_score"])
        }
    
    def _classify_realism_level(self, score: float) -> str:
        """Classify realism level based on score."""
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "moderate"
        else:
            return "low"


def create_realism_analyzer(config: Dict[str, Any]) -> RealismAnalyzer:
    """Factory function to create a RealismAnalyzer."""
    return RealismAnalyzer(config)