"""
Data models and schemas for the synthetic review generator.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json


class ExperienceLevel(Enum):
    """User experience levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


class ToneType(Enum):
    """Review tone types."""
    PROFESSIONAL = "professional"
    BUSINESS_FOCUSED = "business-focused"
    ANALYTICAL = "analytical"
    STRATEGIC = "strategic"
    CRITICAL = "critical"
    RESULTS_FOCUSED = "results-focused"
    ENTHUSIASTIC = "enthusiastic"


class SentimentType(Enum):
    """Review sentiment types."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


@dataclass
class Persona:
    """User persona for review generation."""
    role: str
    experience: ExperienceLevel
    tone: ToneType
    characteristics: List[str]
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary."""
        return {
            "role": self.role,
            "experience": self.experience.value,
            "tone": self.tone.value,
            "characteristics": self.characteristics,
            "weight": self.weight
        }


@dataclass
class QualityScores:
    """Quality scores for a generated review."""
    # Diversity scores
    jaccard_similarity: float = 0.0
    semantic_similarity: float = 0.0
    
    # Bias scores
    sentiment_appropriateness: float = 0.0
    rating_consistency: float = 0.0
    persona_alignment: float = 0.0
    
    # Realism scores
    domain_relevance: float = 0.0
    specificity_score: float = 0.0
    marketing_language_score: float = 0.0
    
    # Overall scores
    composite_score: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert quality scores to dictionary."""
        return asdict(self)
    
    def calculate_composite_score(self) -> float:
        """Calculate overall composite quality score."""
        # Weighted combination of different quality aspects
        diversity_score = ((1 - self.jaccard_similarity) * 0.5 + 
                          (1 - self.semantic_similarity) * 0.5)
        
        bias_score = (self.sentiment_appropriateness * 0.4 + 
                     self.rating_consistency * 0.3 + 
                     self.persona_alignment * 0.3)
        
        realism_score = (self.domain_relevance * 0.4 + 
                        self.specificity_score * 0.3 + 
                        (1 - self.marketing_language_score) * 0.3)
        
        self.composite_score = (diversity_score * 0.3 + 
                               bias_score * 0.3 + 
                               realism_score * 0.4)
        
        return self.composite_score


@dataclass
class ModelMetadata:
    """Metadata about the model used for generation."""
    provider: str
    model_name: str
    temperature: float
    max_tokens: int
    generation_time_ms: int
    api_cost_estimate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model metadata to dictionary."""
        return asdict(self)


@dataclass
class Review:
    """A synthetic or real review."""
    id: str
    rating: int  # 1-5 stars
    review_text: str
    timestamp: datetime
    persona: Optional[Persona] = None
    model_metadata: Optional[ModelMetadata] = None
    quality_scores: Optional[QualityScores] = None
    sentiment: Optional[SentimentType] = None
    word_count: int = 0
    is_synthetic: bool = True
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.word_count == 0:
            self.word_count = len(self.review_text.split())
        
        if not self.id:
            # Generate ID based on timestamp and content hash
            content_hash = hash(self.review_text) % 10000
            time_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
            self.id = f"review_{time_str}_{content_hash}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert review to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "rating": self.rating,
            "review_text": self.review_text,
            "timestamp": self.timestamp.isoformat(),
            "word_count": self.word_count,
            "is_synthetic": self.is_synthetic
        }
        
        if self.persona:
            result["persona"] = self.persona.to_dict()
        
        if self.model_metadata:
            result["model_metadata"] = self.model_metadata.to_dict()
        
        if self.quality_scores:
            result["quality_scores"] = self.quality_scores.to_dict()
        
        if self.sentiment:
            result["sentiment"] = self.sentiment.value
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Review':
        """Create Review from dictionary."""
        # Parse timestamp
        timestamp = datetime.fromisoformat(data["timestamp"])
        
        # Parse persona if present
        persona = None
        if "persona" in data and data["persona"]:
            persona_data = data["persona"]
            persona = Persona(
                role=persona_data["role"],
                experience=ExperienceLevel(persona_data["experience"]),
                tone=ToneType(persona_data["tone"]),
                characteristics=persona_data["characteristics"],
                weight=persona_data.get("weight", 1.0)
            )
        
        # Parse model metadata if present
        model_metadata = None
        if "model_metadata" in data and data["model_metadata"]:
            model_data = data["model_metadata"]
            model_metadata = ModelMetadata(**model_data)
        
        # Parse quality scores if present
        quality_scores = None
        if "quality_scores" in data and data["quality_scores"]:
            quality_scores = QualityScores(**data["quality_scores"])
        
        # Parse sentiment if present
        sentiment = None
        if "sentiment" in data and data["sentiment"]:
            sentiment = SentimentType(data["sentiment"])
        
        return cls(
            id=data["id"],
            rating=data["rating"],
            review_text=data["review_text"],
            timestamp=timestamp,
            persona=persona,
            model_metadata=model_metadata,
            quality_scores=quality_scores,
            sentiment=sentiment,
            word_count=data.get("word_count", 0),
            is_synthetic=data.get("is_synthetic", True)
        )


@dataclass
class GenerationStats:
    """Statistics about the generation process."""
    total_attempts: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    regeneration_attempts: int = 0
    total_generation_time_ms: int = 0
    average_generation_time_ms: float = 0.0
    quality_rejections: int = 0
    
    # Per-model stats
    model_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Quality distribution
    quality_score_distribution: Dict[str, int] = field(default_factory=dict)
    
    def add_generation_attempt(self, success: bool, generation_time_ms: int, 
                              model_provider: str, quality_score: Optional[float] = None):
        """Record a generation attempt."""
        self.total_attempts += 1
        self.total_generation_time_ms += generation_time_ms
        
        # Initialize model stats if needed
        if model_provider not in self.model_stats:
            self.model_stats[model_provider] = {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "total_time_ms": 0,
                "average_time_ms": 0.0
            }
        
        # Update model stats
        model_stat = self.model_stats[model_provider]
        model_stat["attempts"] += 1
        model_stat["total_time_ms"] += generation_time_ms
        
        if success:
            self.successful_generations += 1
            model_stat["successes"] += 1
            
            if quality_score is not None:
                # Update quality distribution
                score_bucket = f"{int(quality_score * 10) / 10:.1f}"
                self.quality_score_distribution[score_bucket] = \
                    self.quality_score_distribution.get(score_bucket, 0) + 1
        else:
            self.failed_generations += 1
            model_stat["failures"] += 1
        
        # Update averages
        if self.total_attempts > 0:
            self.average_generation_time_ms = self.total_generation_time_ms / self.total_attempts
        
        if model_stat["attempts"] > 0:
            model_stat["average_time_ms"] = model_stat["total_time_ms"] / model_stat["attempts"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return asdict(self)


@dataclass
class ComparisonMetrics:
    """Metrics comparing synthetic and real reviews."""
    synthetic_count: int = 0
    real_count: int = 0
    
    # Length comparison
    synthetic_avg_length: float = 0.0
    real_avg_length: float = 0.0
    length_similarity_score: float = 0.0
    
    # Rating distribution comparison
    synthetic_rating_dist: Dict[int, float] = field(default_factory=dict)
    real_rating_dist: Dict[int, float] = field(default_factory=dict)
    rating_distribution_similarity: float = 0.0
    
    # Sentiment comparison
    synthetic_sentiment_dist: Dict[str, float] = field(default_factory=dict)
    real_sentiment_dist: Dict[str, float] = field(default_factory=dict)
    sentiment_distribution_similarity: float = 0.0
    
    # Vocabulary metrics
    synthetic_vocabulary_richness: float = 0.0
    real_vocabulary_richness: float = 0.0
    vocabulary_overlap: float = 0.0
    
    # Topic coverage
    synthetic_topic_coverage: Dict[str, float] = field(default_factory=dict)
    real_topic_coverage: Dict[str, float] = field(default_factory=dict)
    topic_coverage_similarity: float = 0.0
    
    # Overall realism score
    overall_realism_score: float = 0.0
    
    def calculate_overall_score(self) -> float:
        """Calculate overall realism score."""
        scores = [
            self.length_similarity_score,
            self.rating_distribution_similarity,
            self.sentiment_distribution_similarity,
            self.vocabulary_overlap,
            self.topic_coverage_similarity
        ]
        
        # Filter out zero scores (not calculated)
        valid_scores = [s for s in scores if s > 0]
        
        if valid_scores:
            self.overall_realism_score = sum(valid_scores) / len(valid_scores)
        
        return self.overall_realism_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


class ReviewDataset:
    """Container for a collection of reviews with metadata."""
    
    def __init__(self):
        self.reviews: List[Review] = []
        self.generation_stats = GenerationStats()
        self.comparison_metrics = ComparisonMetrics()
        self.metadata: Dict[str, Any] = {}
    
    def add_review(self, review: Review):
        """Add a review to the dataset."""
        self.reviews.append(review)
    
    def get_synthetic_reviews(self) -> List[Review]:
        """Get all synthetic reviews."""
        return [r for r in self.reviews if r.is_synthetic]
    
    def save_to_json(self, filename: str):
        """Save dataset to JSON file."""
        data = {
            "reviews": [review.to_dict() for review in self.reviews],
            "generation_stats": self.generation_stats.to_dict(),
            "comparison_metrics": self.comparison_metrics.to_dict(),
            "metadata": self.metadata
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_json(cls, filename: str) -> 'ReviewDataset':
        """Load dataset from JSON file."""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dataset = cls()
        
        # Load reviews
        for review_data in data.get("reviews", []):
            review = Review.from_dict(review_data)
            dataset.add_review(review)
        
        # Load stats and metrics
        if "generation_stats" in data:
            dataset.generation_stats = GenerationStats(**data["generation_stats"])
        
        if "comparison_metrics" in data:
            dataset.comparison_metrics = ComparisonMetrics(**data["comparison_metrics"])
        
        dataset.metadata = data.get("metadata", {})
        
        return dataset