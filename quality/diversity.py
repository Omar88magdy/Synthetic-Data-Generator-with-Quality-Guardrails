"""
Simplified diversity metrics for synthetic reviews.
"""

import re
import numpy as np
from typing import List, Dict, Set, Tuple, Any
from collections import Counter
import logging

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from generator.models import Review


class DiversityAnalyzer:
    """Analyzes diversity in synthetic reviews using Jaccard and semantic similarity."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize sentence transformer if available
        self.sentence_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                # Disable progress bars to avoid cluttering output
                self.sentence_model.show_progress_bar = False
                self.logger.info("Loaded sentence transformer model")
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                # Ensure deterministic inference
                self.sentence_model.eval()
                self.logger.info("Loaded sentence transformer model with deterministic settings")
            except Exception as e:
                self.logger.warning(f"Failed to load sentence transformer: {e}")
                self.sentence_model = None
    
    def calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        tokens1 = set(self._tokenize(text1))
        tokens2 = set(self._tokenize(text2))
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
            
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using sentence transformers."""
        if not self.sentence_model:
            # Fallback to Jaccard similarity if sentence transformer not available
            return self.calculate_jaccard_similarity(text1, text2)
        
        try:
            embeddings = self.sentence_model.encode([text1, text2], show_progress_bar=False)
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            self.logger.warning(f"Semantic similarity calculation failed: {e}")
            return self.calculate_jaccard_similarity(text1, text2)
    
    def analyze_batch_diversity(self, batch_reviews: List[str]) -> Dict[str, Any]:
        """Analyze diversity of a batch of reviews using both Jaccard and semantic similarity."""
        if len(batch_reviews) < 2:
            return {
                "jaccard_diversity": 1.0,
                "semantic_diversity": 1.0,
                "overall_diversity": 1.0,
                "total_reviews": len(batch_reviews)
            }
        
        # Calculate all pairwise similarities
        jaccard_similarities = []
        semantic_similarities = []
        
        for i in range(len(batch_reviews)):
            for j in range(i + 1, len(batch_reviews)):
                jaccard_sim = self.calculate_jaccard_similarity(batch_reviews[i], batch_reviews[j])
                semantic_sim = self.calculate_semantic_similarity(batch_reviews[i], batch_reviews[j])
                
                jaccard_similarities.append(jaccard_sim)
                semantic_similarities.append(semantic_sim)
        
        # Calculate diversity scores (1 - average similarity)
        avg_jaccard_similarity = np.mean(jaccard_similarities)
        avg_semantic_similarity = np.mean(semantic_similarities)
        
        jaccard_diversity = 1.0 - avg_jaccard_similarity
        semantic_diversity = 1.0 - avg_semantic_similarity
        
        # Overall diversity is the average of both metrics
        overall_diversity = (jaccard_diversity + semantic_diversity) / 2.0
        
        return {
            "jaccard_diversity": jaccard_diversity,
            "semantic_diversity": semantic_diversity,
            "overall_diversity": overall_diversity,
            "avg_jaccard_similarity": avg_jaccard_similarity,
            "avg_semantic_similarity": avg_semantic_similarity,
            "total_reviews": len(batch_reviews),
            "total_comparisons": len(jaccard_similarities)
        }
    
    def analyze_diversity(self, reviews: List[str]) -> Dict[str, Any]:
        """Main method to analyze diversity - calls analyze_batch_diversity."""
        return self.analyze_batch_diversity(reviews)
    
    def check_diversity_against_corpus(self, new_review: Review, existing_reviews: List[Review]) -> Dict[str, Any]:
        """Check diversity of a new review against existing corpus (deterministic)."""
        if not existing_reviews:
            return {
                "diversity_score": 1.0,
                "passes_threshold": True,
                "max_jaccard_similarity": 0.0,
                "max_semantic_similarity": 0.0
            }
        
        new_text = new_review.review_text
        # Sort existing reviews by ID for deterministic order
        sorted_existing = sorted(existing_reviews, key=lambda r: r.id or "")
        existing_texts = [review.review_text for review in sorted_existing]
        
        # Calculate similarities against all existing reviews
        jaccard_similarities = []
        semantic_similarities = []
        
        for existing_text in existing_texts:
            jaccard_sim = self.calculate_jaccard_similarity(new_text, existing_text)
            semantic_sim = self.calculate_semantic_similarity(new_text, existing_text)
            
            jaccard_similarities.append(jaccard_sim)
            semantic_similarities.append(semantic_sim)
        
        # Find the highest similarities with consistent precision
        max_jaccard = round(max(jaccard_similarities), 6) if jaccard_similarities else 0.0
        max_semantic = round(max(semantic_similarities), 6) if semantic_similarities else 0.0
        
        # Overall diversity score (1 - highest similarity)
        diversity_score = 1.0 - max(max_jaccard, max_semantic)
        
        # Simple threshold check
        threshold = 0.8
        passes_threshold = max(max_jaccard, max_semantic) <= threshold
        
        return {
            "diversity_score": diversity_score,
            "max_jaccard_similarity": max_jaccard,
            "max_semantic_similarity": max_semantic,
            "passes_threshold": passes_threshold,
            "corpus_size": len(existing_reviews)
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for text analysis with stop word removal."""
        # Common English stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to',
            'was', 'will', 'with', 'i', 'you', 'we', 'they', 'she', 'her', 'his',
            'him', 'me', 'my', 'our', 'your', 'their', 'this', 'that', 'these',
            'those', 'am', 'can', 'could', 'do', 'does', 'did', 'have', 'had',
            'but', 'or', 'not', 'no', 'yes', 'so', 'if', 'then', 'than', 'when',
            'where', 'why', 'how', 'what', 'which', 'who', 'would', 'should',
            'may', 'might', 'must', 'shall', 'very', 'much', 'more', 'most'
        }
        
        # Convert to lowercase and extract words
        text = text.lower()
        # Remove punctuation and split
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        # Filter out stop words and short words
        return [word for word in words if len(word) > 2 and word not in stop_words]


def create_diversity_analyzer(config: Dict[str, Any]) -> DiversityAnalyzer:
    """Factory function to create a DiversityAnalyzer."""
    return DiversityAnalyzer(config)