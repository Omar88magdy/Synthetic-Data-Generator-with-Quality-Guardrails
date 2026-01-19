"""
Quality scoring and regeneration management for synthetic reviews.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .models import Review, QualityScores
from quality.diversity import DiversityAnalyzer
from quality.bias import BiasDetector
from quality.realism import RealismAnalyzer


logger = logging.getLogger(__name__)


class QualityScorer:
    """Scores the quality of generated reviews using multiple metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.diversity_analyzer = DiversityAnalyzer(config)
        self.bias_detector = BiasDetector(config)
        self.realism_analyzer = RealismAnalyzer(config)
        real_reviews_path = config.get('real_reviews_path', 'data/easy_generator_real_reviews.json')
        self.real_reviews = self._load_real_reviews(real_reviews_path)
    
    def _calculate_overall_score(self, diversity: float, bias: float, realism: float) -> float:
        """Calculate overall quality score."""
        # Weight the scores (lower bias is better)
        weighted_score = (
            diversity * 0.3 +
            (1.0 - bias) * 0.3 +  # Invert bias score
            realism * 0.4
        )
        return max(0.0, min(1.0, weighted_score))


class RegenerationManager:
    """Manages the regeneration of low-quality reviews."""
    
    def __init__(self, quality_threshold: float = 0.6):
        self.quality_threshold = quality_threshold
        self.regeneration_history: List[Dict[str, Any]] = []
    
    def get_regeneration_stats(self) -> Dict[str, Any]:
        """Get statistics about regeneration attempts."""
        if not self.regeneration_history:
            return {'total_attempts': 0}
        
        improvements = [entry['improvement'] for entry in self.regeneration_history]
        
        return {
            'total_attempts': len(self.regeneration_history),
            'average_improvement': sum(improvements) / len(improvements),
            'successful_improvements': len([i for i in improvements if i > 0]),
            'improvement_rate': len([i for i in improvements if i > 0]) / len(improvements)
        }