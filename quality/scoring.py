"""
Quality scoring and automatic regeneration system.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from generator.models import Review, QualityScores
from .diversity import DiversityAnalyzer
from .bias import BiasDetector  
from .realism import RealismAnalyzer


@dataclass
class RegenerationAttempt:
    """Records an attempt at regeneration."""
    attempt_number: int
    reason: str
    original_score: float
    new_score: Optional[float] = None
    success: bool = False
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class QualityScorer:
    """Calculates comprehensive quality scores for reviews."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = config.get("quality_thresholds", {})
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize analyzers
        self.diversity_analyzer = DiversityAnalyzer(config)
        self.bias_detector = BiasDetector(config)
        self.realism_analyzer = RealismAnalyzer(config)
    
    def score_review(self, review: Review, existing_reviews: List[Review]) -> QualityScores:
        """Calculate comprehensive quality scores for a review."""
        scores = QualityScores()
        
        # Handle None review case
        if review is None:
            self.logger.error("Cannot score None review")
            scores.composite_score = 0.0
            return scores
        
        try:
            # Diversity analysis
            diversity_result = self.diversity_analyzer.check_diversity_against_corpus(
                review, existing_reviews
            )
            
            if diversity_result:
                scores.jaccard_similarity = diversity_result.get("max_jaccard_similarity", 0.0)
                scores.semantic_similarity = diversity_result.get("max_semantic_similarity", 0.0)
            
            # Realism analysis
            realism_result = self.realism_analyzer.calculate_overall_realism_score(review)
            
            if realism_result:
                # Get domain relevance directly from the result
                scores.domain_relevance = realism_result.get("domain_relevance", 0.0)
                
                # For now, set specificity and marketing scores to reasonable defaults
                # since our simplified realism analyzer doesn't compute these
                scores.specificity_score = 0.5  # Neutral score instead of 0
                scores.marketing_language_score = 0.0  # No marketing language detected (good)
            
            # Bias analysis (individual review level)
            scores.sentiment_appropriateness = self._calculate_sentiment_appropriateness(review)
            scores.rating_consistency = self._calculate_rating_consistency(review)
            scores.persona_alignment = self._calculate_persona_alignment(review)
            
            # Calculate composite score
            scores.calculate_composite_score()
            
            self.logger.debug(f"Scored review {review.id}: {scores.composite_score:.3f}")
            
        except Exception as e:
            review_id = getattr(review, 'id', 'unknown')
            self.logger.error(f"Error scoring review {review_id}: {e}")
            # Return default low scores on error
            scores.composite_score = 0.0
        
        return scores
    
    def _calculate_sentiment_appropriateness(self, review: Review) -> float:
        """Calculate how appropriate the sentiment is for the rating."""
        if not review.sentiment:
            return 0.5  # Neutral score for unknown sentiment
        
        rating = review.rating
        sentiment = review.sentiment.value
        
        # Expected sentiment ranges for ratings
        if rating >= 4:
            expected_sentiment = "positive"
        elif rating <= 2:
            expected_sentiment = "negative"
        else:
            expected_sentiment = "neutral"
        
        # Score based on alignment
        if sentiment == expected_sentiment:
            return 1.0
        elif (sentiment == "neutral" and rating == 3) or \
             (sentiment == "positive" and rating >= 3) or \
             (sentiment == "negative" and rating <= 3):
            return 0.7  # Acceptable alignment
        else:
            return 0.3  # Poor alignment
    
    def _calculate_rating_consistency(self, review: Review) -> float:
        """Calculate how consistent the rating is with expected distribution."""
        expected_dist = self.config.get("rating_distribution", {})
        rating = review.rating
        
        # Higher expected frequency = higher consistency score
        expected_freq = expected_dist.get(rating, 0.2)
        
        # Convert frequency to consistency score (0.05-1.0 -> 0.3-1.0)
        return min(1.0, max(0.3, expected_freq * 5))
    
    def _calculate_persona_alignment(self, review: Review) -> float:
        """Calculate how well the review aligns with the persona."""
        if not review.persona:
            return 0.5
        
        # This is a simplified implementation
        # In practice, you might analyze language patterns, technical depth, etc.
        
        persona = review.persona
        rating = review.rating
        
        # Some personas might be more critical or positive
        if persona.tone.value == "critical" and rating >= 4:
            return 0.6  # Critical personas giving high ratings should be less common
        elif persona.tone.value == "enthusiastic" and rating <= 2:
            return 0.6  # Enthusiastic personas giving low ratings should be less common
        elif persona.experience.value == "expert" and "beginner" in review.review_text.lower():
            return 0.4  # Experts shouldn't talk like beginners
        else:
            return 0.8  # Default good alignment
    
    def passes_quality_threshold(self, scores: QualityScores, existing_reviews: List[Review] = None) -> Tuple[bool, List[str]]:
        """Check if quality scores pass all thresholds with progressive scaling."""
        failures = []
        
        # Calculate progressive thresholds based on corpus size
        corpus_size = len(existing_reviews) if existing_reviews else 0
        progressive_thresholds = self._calculate_progressive_thresholds(corpus_size)
        
        min_quality_score = self.thresholds.get("min_quality_score", 0.6)
        if scores.composite_score < min_quality_score:
            failures.append(f"Composite score {scores.composite_score:.3f} < {min_quality_score}")
        
        max_jaccard_similarity = progressive_thresholds["max_jaccard_similarity"]
        if scores.jaccard_similarity > max_jaccard_similarity:
            failures.append(f"Jaccard similarity {scores.jaccard_similarity:.3f} > {max_jaccard_similarity:.3f} (progressive)")
        
        max_semantic_similarity = progressive_thresholds["max_semantic_similarity"]
        if scores.semantic_similarity > max_semantic_similarity:
            failures.append(f"Semantic similarity {scores.semantic_similarity:.3f} > {max_semantic_similarity:.3f} (progressive)")
        

        
        max_marketing_score = self.thresholds.get("max_marketing_score", 0.3)
        if scores.marketing_language_score > max_marketing_score:
            failures.append(f"Marketing language {scores.marketing_language_score:.3f} > {max_marketing_score}")
        
        min_specificity_score = self.thresholds.get("min_specificity_score", 0.4)
        if scores.specificity_score < min_specificity_score:
            failures.append(f"Specificity score {scores.specificity_score:.3f} < {min_specificity_score}")
        
        return len(failures) == 0, failures
    
    def _calculate_progressive_thresholds(self, corpus_size: int) -> Dict[str, float]:
        """Calculate progressive thresholds that relax as corpus grows."""
        # Base thresholds from config
        base_jaccard = self.thresholds.get("max_jaccard_similarity", 0.45)
        base_semantic = self.thresholds.get("max_semantic_similarity", 0.85)
        
        # Progressive scaling based on corpus size
        # Scale factor increases from 1.0 to 1.5 as corpus grows from 0 to 200
        scale_factor = 1.0 + (corpus_size / 400)  # 0 -> 1.0, 200 -> 1.5
        
        return {
            "max_jaccard_similarity": min(0.7, base_jaccard * scale_factor),
            "max_semantic_similarity": min(0.95, base_semantic * scale_factor)
        }


class RegenerationManager:
    """Manages automatic regeneration of low-quality reviews."""
    
    def __init__(self, config: Dict[str, Any], review_generator):
        self.config = config
        self.regeneration_config = config.get("regeneration", {})
        self.max_attempts = self.regeneration_config.get("max_attempts", 5)
        self.cooldown_seconds = self.regeneration_config.get("cooldown_seconds", 1)
        self.review_generator = review_generator
        self.quality_scorer = QualityScorer(config)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Track regeneration statistics
        self.regeneration_stats = {
            "total_regenerations": 0,
            "successful_regenerations": 0,
            "failed_regenerations": 0,
            "reasons": {},
            "attempts_distribution": {}
        }
    
    def regenerate_if_needed(self, review: Review, existing_reviews: List[Review]) -> Tuple[Review, List[RegenerationAttempt]]:
        """Regenerate review if it fails quality checks."""
        attempts = []
        current_review = review
        
        # Score the initial review
        quality_scores = self.quality_scorer.score_review(current_review, existing_reviews)
        current_review.quality_scores = quality_scores
        
        passes, failures = self.quality_scorer.passes_quality_threshold(quality_scores)
        
        if passes:
            self.logger.debug(f"Review {review.id} passes quality checks")
            return current_review, attempts
        
        self.logger.info(f"Review {review.id} failed quality checks: {failures}")
        
        # Attempt regeneration
        for attempt_num in range(1, self.max_attempts + 1):
            self.logger.info(f"Regeneration attempt {attempt_num}/{self.max_attempts} for review {review.id}")
            
            attempt = RegenerationAttempt(
                attempt_number=attempt_num,
                reason="; ".join(failures),
                original_score=quality_scores.composite_score
            )
            
            try:
                # Wait for cooldown
                if self.cooldown_seconds > 0:
                    time.sleep(self.cooldown_seconds)
                
                # Generate new review with same parameters
                new_review = self._regenerate_review(current_review)
                
                if new_review:
                    # Score the new review
                    new_quality_scores = self.quality_scorer.score_review(new_review, existing_reviews)
                    new_review.quality_scores = new_quality_scores
                    
                    new_passes, new_failures = self.quality_scorer.passes_quality_threshold(new_quality_scores)
                    
                    attempt.new_score = new_quality_scores.composite_score
                    attempt.success = new_passes
                    
                    if new_passes:
                        self.logger.info(f"Regeneration successful on attempt {attempt_num}")
                        current_review = new_review
                        self._update_regeneration_stats(attempt_num, failures[0], True)
                        attempts.append(attempt)
                        break
                    else:
                        self.logger.debug(f"Regeneration attempt {attempt_num} still failed: {new_failures}")
                        current_review = new_review  # Keep the better one
                        failures = new_failures
                        quality_scores = new_quality_scores
                
            except Exception as e:
                self.logger.error(f"Regeneration attempt {attempt_num} failed with error: {e}")
                attempt.success = False
            
            attempts.append(attempt)
        
        # Update stats
        if not attempts or not attempts[-1].success:
            self._update_regeneration_stats(len(attempts), failures[0] if failures else "unknown", False)
            self.logger.warning(f"All regeneration attempts failed for review {review.id}")
        
        return current_review, attempts
    
    def _regenerate_review(self, original_review: Review) -> Optional[Review]:
        """Generate a new review with the same parameters as the original."""
        try:
            # Use the same persona and rating as the original
            provider_name = original_review.model_metadata.provider if original_review.model_metadata else None
            
            if provider_name and provider_name in self.review_generator.model_providers:
                # Generate with the same provider to maintain consistency
                provider_distribution = {provider_name: 1.0}
            else:
                # Fall back to random provider selection
                provider_weights = {name: config.get("weight", 1.0) 
                                  for name, config in self.config["models"].items() 
                                  if config.get("enabled", False)}
                total_weight = sum(provider_weights.values())
                provider_distribution = {name: weight / total_weight 
                                       for name, weight in provider_weights.items()}
            
            # Generate new review (this is a simplified version - in practice you'd call the review generator)
            new_review = self.review_generator._generate_single_review(provider_distribution)
            
            # Ensure same persona and rating
            if new_review and original_review.persona:
                new_review.persona = original_review.persona
                new_review.rating = original_review.rating
            
            return new_review
            
        except Exception as e:
            self.logger.error(f"Failed to regenerate review: {e}")
            return None
    
    def _update_regeneration_stats(self, attempts: int, reason: str, success: bool):
        """Update regeneration statistics."""
        self.regeneration_stats["total_regenerations"] += 1
        
        if success:
            self.regeneration_stats["successful_regenerations"] += 1
        else:
            self.regeneration_stats["failed_regenerations"] += 1
        
        # Track reasons
        reason_category = reason.split()[0]  # First word of reason
        self.regeneration_stats["reasons"][reason_category] = \
            self.regeneration_stats["reasons"].get(reason_category, 0) + 1
        
        # Track attempt distribution
        self.regeneration_stats["attempts_distribution"][attempts] = \
            self.regeneration_stats["attempts_distribution"].get(attempts, 0) + 1
    
    def get_regeneration_stats(self) -> Dict[str, Any]:
        """Get regeneration statistics."""
        stats = self.regeneration_stats.copy()
        
        if stats["total_regenerations"] > 0:
            stats["success_rate"] = stats["successful_regenerations"] / stats["total_regenerations"]
            stats["average_attempts"] = sum(
                attempts * count for attempts, count in stats["attempts_distribution"].items()
            ) / stats["total_regenerations"]
        else:
            stats["success_rate"] = 0.0
            stats["average_attempts"] = 0.0
        
        return stats


def create_quality_scorer(config: Dict[str, Any]) -> QualityScorer:
    """Factory function to create a quality scorer."""
    return QualityScorer(config)