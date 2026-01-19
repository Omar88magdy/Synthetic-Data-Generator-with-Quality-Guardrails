"""
Comparison logic for analyzing synthetic vs real reviews.
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter
import logging
from datetime import datetime

from generator.models import Review, ReviewDataset, ComparisonMetrics, SentimentType


class ReviewComparator:
    """Compares synthetic and real reviews across multiple dimensions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.comparison_config = config.get("comparison", {})
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_real_reviews(self, file_path: str) -> List[Review]:
        """Load real reviews from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            real_reviews = []
            for item in data:
                # Extract review components from the real data structure
                content = item.get("content", {})
                likes = content.get("likes", "")
                dislikes = content.get("dislikes", "")
                benefits = content.get("benefits", "")
                
                # Combine all text content into review_text
                review_parts = []
                if likes:
                    review_parts.append(f"Likes: {likes}")
                if dislikes:
                    review_parts.append(f"Dislikes: {dislikes}")
                if benefits:
                    review_parts.append(f"Benefits: {benefits}")
                
                review_text = " ".join(review_parts)
                
                # Create a simple ID if not present
                review_id = item.get("id", f"real_review_{len(real_reviews)}")
                
                # Get rating from review section
                rating = item.get("review", {}).get("rating", 5)
                
                # Get date from reviewer section
                date_str = item.get("reviewer", {}).get("date", "2025-01-01")
                
                # Convert to Review object
                review = Review(
                    id=review_id,
                    rating=int(rating),
                    review_text=review_text,
                    timestamp=datetime.fromisoformat(date_str + "T00:00:00"),
                    sentiment=SentimentType("neutral"),  # Default sentiment
                    word_count=len(review_text.split()),
                    is_synthetic=False
                )
                real_reviews.append(review)
            
            self.logger.info(f"Loaded {len(real_reviews)} real reviews")
            return real_reviews
            
        except Exception as e:
            self.logger.error(f"Failed to load real reviews: {e}")
            return []
    
    def compare_length_distributions(self, synthetic_reviews: List[Review], 
                                   real_reviews: List[Review]) -> Dict[str, Any]:
        """Compare length distributions between synthetic and real reviews."""
        synthetic_lengths = [r.word_count for r in synthetic_reviews]
        real_lengths = [r.word_count for r in real_reviews]
        
        result = {
            "synthetic_stats": self._calculate_stats(synthetic_lengths),
            "real_stats": self._calculate_stats(real_lengths),
            "similarity_score": 0.0,
            "distribution_comparison": {}
        }
        
        # Calculate similarity score using statistical comparison
        if synthetic_lengths and real_lengths:
            # Use Kolmogorov-Smirnov test for distribution similarity
            try:
                from scipy import stats
                ks_statistic, p_value = stats.ks_2samp(synthetic_lengths, real_lengths)
                # Convert KS statistic to similarity score (1 - KS statistic)
                result["similarity_score"] = max(0, 1 - ks_statistic)
                result["ks_test"] = {
                    "statistic": float(ks_statistic),
                    "p_value": float(p_value),
                    "significant_difference": p_value < 0.05
                }
            except ImportError:
                # Fallback: simple mean comparison
                syn_mean = np.mean(synthetic_lengths)
                real_mean = np.mean(real_lengths)
                mean_diff = abs(syn_mean - real_mean) / max(syn_mean, real_mean)
                result["similarity_score"] = max(0, 1 - mean_diff)
        
        # Length category distribution
        length_categories = {"short": (0, 100), "medium": (100, 200), "long": (200, float('inf'))}
        
        syn_categories = self._categorize_lengths(synthetic_lengths, length_categories)
        real_categories = self._categorize_lengths(real_lengths, length_categories)
        
        result["distribution_comparison"] = {
            "synthetic_categories": syn_categories,
            "real_categories": real_categories,
            "category_similarity": self._calculate_distribution_similarity(syn_categories, real_categories)
        }
        
        return result
    
    def compare_rating_distributions(self, synthetic_reviews: List[Review], 
                                   real_reviews: List[Review]) -> Dict[str, Any]:
        """Compare rating distributions between synthetic and real reviews."""
        synthetic_ratings = [r.rating for r in synthetic_reviews]
        real_ratings = [r.rating for r in real_reviews]
        
        # Calculate distributions
        syn_dist = self._calculate_rating_distribution(synthetic_ratings)
        real_dist = self._calculate_rating_distribution(real_ratings)
        
        # Calculate similarity
        similarity = self._calculate_distribution_similarity(syn_dist, real_dist)
        
        result = {
            "synthetic_distribution": syn_dist,
            "real_distribution": real_dist,
            "similarity_score": similarity,
            "average_ratings": {
                "synthetic": float(np.mean(synthetic_ratings)) if synthetic_ratings else 0,
                "real": float(np.mean(real_ratings)) if real_ratings else 0
            }
        }
        
        # Statistical comparison
        if synthetic_ratings and real_ratings:
            try:
                from scipy import stats
                # Chi-square test for rating distribution similarity
                syn_counts = [syn_dist.get(i, 0) * len(synthetic_ratings) for i in range(1, 6)]
                real_counts = [real_dist.get(i, 0) * len(real_ratings) for i in range(1, 6)]
                
                # Check if we have valid data for chi-square test
                if sum(syn_counts) > 0 and sum(real_counts) > 0:
                    # Add small constant to avoid zeros
                    syn_counts = [max(count, 0.01) for count in syn_counts]
                    real_counts = [max(count, 0.01) for count in real_counts]
                    
                    # Normalize to same total for comparison
                    syn_total = sum(syn_counts)
                    real_total = sum(real_counts)
                    syn_counts = [count / syn_total * min(syn_total, real_total) for count in syn_counts]
                    real_counts = [count / real_total * min(syn_total, real_total) for count in real_counts]
                    
                    chi2_statistic, p_value = stats.chisquare(syn_counts, real_counts)
                    result["chi_square_test"] = {
                        "statistic": float(chi2_statistic),
                        "p_value": float(p_value),
                        "significant_difference": p_value < 0.05
                    }
            except (ImportError, ValueError) as e:
                self.logger.error(f"Error during comparison: {e}")
                pass
        
        return result
    
    def compare_sentiment_distributions(self, synthetic_reviews: List[Review], 
                                      real_reviews: List[Review]) -> Dict[str, Any]:
        """Compare sentiment distributions between synthetic and real reviews."""
        synthetic_sentiments = [r.sentiment.value if r.sentiment else "unknown" for r in synthetic_reviews]
        real_sentiments = [r.sentiment.value if r.sentiment else "unknown" for r in real_reviews]
        
        # Calculate distributions
        syn_dist = self._calculate_sentiment_distribution(synthetic_sentiments)
        real_dist = self._calculate_sentiment_distribution(real_sentiments)
        
        # Calculate similarity
        similarity = self._calculate_distribution_similarity(syn_dist, real_dist)
        
        return {
            "synthetic_distribution": syn_dist,
            "real_distribution": real_dist,
            "similarity_score": similarity
        }
    
    def analyze_vocabulary_richness(self, synthetic_reviews: List[Review], 
                                  real_reviews: List[Review]) -> Dict[str, Any]:
        """Analyze vocabulary richness and overlap."""
        synthetic_texts = [r.review_text for r in synthetic_reviews]
        real_texts = [r.review_text for r in real_reviews]
        
        # Extract vocabulary
        synthetic_vocab = self._extract_vocabulary(synthetic_texts)
        real_vocab = self._extract_vocabulary(real_texts)
        
        # Calculate metrics
        vocab_overlap = len(synthetic_vocab.intersection(real_vocab))
        vocab_union = len(synthetic_vocab.union(real_vocab))
        
        jaccard_similarity = vocab_overlap / vocab_union if vocab_union > 0 else 0
        
        # Calculate vocabulary richness (unique words / total words)
        syn_all_words = []
        real_all_words = []
        
        for text in synthetic_texts:
            syn_all_words.extend(self._tokenize(text))
        
        for text in real_texts:
            real_all_words.extend(self._tokenize(text))
        
        syn_richness = len(synthetic_vocab) / len(syn_all_words) if syn_all_words else 0
        real_richness = len(real_vocab) / len(real_all_words) if real_all_words else 0
        
        return {
            "synthetic_vocabulary_size": len(synthetic_vocab),
            "real_vocabulary_size": len(real_vocab),
            "vocabulary_overlap": vocab_overlap,
            "vocabulary_union": vocab_union,
            "jaccard_similarity": jaccard_similarity,
            "synthetic_richness": syn_richness,
            "real_richness": real_richness,
            "richness_similarity": 1 - abs(syn_richness - real_richness)
        }
    
    def analyze_topic_coverage(self, synthetic_reviews: List[Review], 
                             real_reviews: List[Review]) -> Dict[str, Any]:
        """Analyze topic coverage using domain keywords."""
        domain_keywords = self.config.get("domain_keywords", [])
        
        synthetic_texts = [r.review_text.lower() for r in synthetic_reviews]
        real_texts = [r.review_text.lower() for r in real_reviews]
        
        # Calculate keyword coverage
        syn_coverage = self._calculate_keyword_coverage(synthetic_texts, domain_keywords)
        real_coverage = self._calculate_keyword_coverage(real_texts, domain_keywords)
        
        # Calculate similarity
        coverage_similarity = self._calculate_distribution_similarity(syn_coverage, real_coverage)
        
        return {
            "synthetic_topic_coverage": syn_coverage,
            "real_topic_coverage": real_coverage,
            "coverage_similarity": coverage_similarity,
            "total_keywords": len(domain_keywords)
        }
    
    def perform_comprehensive_comparison(self, synthetic_reviews: List[Review], 
                                       real_reviews: List[Review]) -> ComparisonMetrics:
        """Perform comprehensive comparison across all dimensions."""
        metrics = ComparisonMetrics()
        
        metrics.synthetic_count = len(synthetic_reviews)
        metrics.real_count = len(real_reviews)
        
        if not synthetic_reviews or not real_reviews:
            self.logger.warning("Insufficient data for comparison")
            return metrics
        
        try:
            # Length comparison
            length_comparison = self.compare_length_distributions(synthetic_reviews, real_reviews)
            metrics.synthetic_avg_length = length_comparison["synthetic_stats"]["mean"]
            metrics.real_avg_length = length_comparison["real_stats"]["mean"]
            metrics.length_similarity_score = length_comparison["similarity_score"]
            
            # Rating comparison
            rating_comparison = self.compare_rating_distributions(synthetic_reviews, real_reviews)
            metrics.synthetic_rating_dist = rating_comparison["synthetic_distribution"]
            metrics.real_rating_dist = rating_comparison["real_distribution"]
            metrics.rating_distribution_similarity = rating_comparison["similarity_score"]
            
            # Sentiment comparison
            sentiment_comparison = self.compare_sentiment_distributions(synthetic_reviews, real_reviews)
            metrics.synthetic_sentiment_dist = sentiment_comparison["synthetic_distribution"]
            metrics.real_sentiment_dist = sentiment_comparison["real_distribution"]
            metrics.sentiment_distribution_similarity = sentiment_comparison["similarity_score"]
            
            # Vocabulary analysis
            vocab_analysis = self.analyze_vocabulary_richness(synthetic_reviews, real_reviews)
            metrics.synthetic_vocabulary_richness = vocab_analysis["synthetic_richness"]
            metrics.real_vocabulary_richness = vocab_analysis["real_richness"]
            metrics.vocabulary_overlap = vocab_analysis["jaccard_similarity"]
            
            # Topic coverage
            topic_analysis = self.analyze_topic_coverage(synthetic_reviews, real_reviews)
            metrics.synthetic_topic_coverage = topic_analysis["synthetic_topic_coverage"]
            metrics.real_topic_coverage = topic_analysis["real_topic_coverage"]
            metrics.topic_coverage_similarity = topic_analysis["coverage_similarity"]
            
            # Calculate overall realism score
            metrics.calculate_overall_score()
            
            self.logger.info(f"Comparison completed. Overall realism score: {metrics.overall_realism_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error during comparison: {e}")
        
        return metrics
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic statistics for a list of values."""
        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
        
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values))
        }
    
    def _categorize_lengths(self, lengths: List[int], categories: Dict[str, Tuple[int, float]]) -> Dict[str, float]:
        """Categorize lengths into predefined categories."""
        categorized = {cat: 0 for cat in categories}
        
        for length in lengths:
            for cat, (min_len, max_len) in categories.items():
                if min_len <= length < max_len:
                    categorized[cat] += 1
                    break
        
        total = len(lengths)
        return {cat: count / total for cat, count in categorized.items()} if total > 0 else categorized
    
    def _calculate_rating_distribution(self, ratings: List[int]) -> Dict[int, float]:
        """Calculate rating distribution."""
        if not ratings:
            return {}
        
        rating_counts = Counter(ratings)
        total = len(ratings)
        
        return {rating: count / total for rating, count in rating_counts.items()}
    
    def _calculate_sentiment_distribution(self, sentiments: List[str]) -> Dict[str, float]:
        """Calculate sentiment distribution."""
        if not sentiments:
            return {}
        
        sentiment_counts = Counter(sentiments)
        total = len(sentiments)
        
        return {sentiment: count / total for sentiment, count in sentiment_counts.items()}
    
    def _calculate_distribution_similarity(self, dist1: Dict[Any, float], dist2: Dict[Any, float]) -> float:
        """Calculate similarity between two probability distributions using Jensen-Shannon divergence."""
        if not dist1 or not dist2:
            return 0.0
        
        # Get all keys
        all_keys = set(dist1.keys()) | set(dist2.keys())
        
        # Convert to probability vectors
        vec1 = np.array([dist1.get(key, 0) for key in all_keys])
        vec2 = np.array([dist2.get(key, 0) for key in all_keys])
        
        # Normalize to ensure they're probability distributions
        vec1 = vec1 / np.sum(vec1) if np.sum(vec1) > 0 else vec1
        vec2 = vec2 / np.sum(vec2) if np.sum(vec2) > 0 else vec2
        
        # Calculate Jensen-Shannon divergence
        def kl_divergence(p, q):
            epsilon = 1e-10  # Small value to avoid log(0)
            return np.sum(p * np.log((p + epsilon) / (q + epsilon)))
        
        m = 0.5 * (vec1 + vec2)
        js_divergence = 0.5 * kl_divergence(vec1, m) + 0.5 * kl_divergence(vec2, m)
        
        # Convert divergence to similarity (0 = identical, higher = more different)
        # JS divergence is in [0, ln(2)], so we normalize and convert to similarity
        similarity = max(0, 1 - js_divergence / np.log(2))
        
        return float(similarity)
    
    def _extract_vocabulary(self, texts: List[str]) -> set:
        """Extract vocabulary from a list of texts."""
        vocab = set()
        for text in texts:
            words = self._tokenize(text)
            vocab.update(words)
        return vocab
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        # Convert to lowercase and extract words
        text = text.lower()
        # Remove punctuation and split into words
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        # Filter out very short words
        return [word for word in words if len(word) > 2]
    
    def _calculate_keyword_coverage(self, texts: List[str], keywords: List[str]) -> Dict[str, float]:
        """Calculate keyword coverage across texts."""
        keyword_counts = {keyword.lower(): 0 for keyword in keywords}
        total_texts = len(texts)
        
        if total_texts == 0:
            return keyword_counts
        
        for text in texts:
            text_lower = text.lower()
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    keyword_counts[keyword.lower()] += 1
        
        # Convert to frequencies
        return {keyword: count / total_texts for keyword, count in keyword_counts.items()}
    
    def generate_comparison_report(self, metrics: ComparisonMetrics) -> str:
        """Generate a detailed comparison report."""
        report_lines = []
        report_lines.append("# Synthetic vs Real Review Comparison Report")
        report_lines.append("")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary
        report_lines.append("## Summary")
        report_lines.append(f"- Synthetic reviews: {metrics.synthetic_count}")
        report_lines.append(f"- Real reviews: {metrics.real_count}")
        report_lines.append(f"- **Overall realism score: {metrics.overall_realism_score:.3f}**")
        report_lines.append("")
        
        # Length comparison
        report_lines.append("## Length Comparison")
        report_lines.append(f"- Synthetic average length: {metrics.synthetic_avg_length:.1f} words")
        report_lines.append(f"- Real average length: {metrics.real_avg_length:.1f} words")
        report_lines.append(f"- Length similarity: {metrics.length_similarity_score:.3f}")
        report_lines.append("")
        
        # Rating distribution
        report_lines.append("## Rating Distribution")
        if metrics.synthetic_rating_dist and metrics.real_rating_dist:
            report_lines.append("| Rating | Synthetic | Real |")
            report_lines.append("|--------|-----------|------|")
            for rating in range(1, 6):
                syn_pct = metrics.synthetic_rating_dist.get(rating, 0) * 100
                real_pct = metrics.real_rating_dist.get(rating, 0) * 100
                report_lines.append(f"| {rating} | {syn_pct:.1f}% | {real_pct:.1f}% |")
            
            report_lines.append(f"\nRating distribution similarity: {metrics.rating_distribution_similarity:.3f}")
        report_lines.append("")
        
        # Sentiment distribution
        report_lines.append("## Sentiment Distribution")
        if metrics.synthetic_sentiment_dist and metrics.real_sentiment_dist:
            report_lines.append("| Sentiment | Synthetic | Real |")
            report_lines.append("|-----------|-----------|------|")
            sentiments = set(metrics.synthetic_sentiment_dist.keys()) | set(metrics.real_sentiment_dist.keys())
            for sentiment in sentiments:
                syn_pct = metrics.synthetic_sentiment_dist.get(sentiment, 0) * 100
                real_pct = metrics.real_sentiment_dist.get(sentiment, 0) * 100
                report_lines.append(f"| {sentiment} | {syn_pct:.1f}% | {real_pct:.1f}% |")
            
            report_lines.append(f"\nSentiment distribution similarity: {metrics.sentiment_distribution_similarity:.3f}")
        report_lines.append("")
        
        # Vocabulary analysis
        report_lines.append("## Vocabulary Analysis")
        report_lines.append(f"- Synthetic vocabulary richness: {metrics.synthetic_vocabulary_richness:.3f}")
        report_lines.append(f"- Real vocabulary richness: {metrics.real_vocabulary_richness:.3f}")
        report_lines.append(f"- Vocabulary overlap (Jaccard): {metrics.vocabulary_overlap:.3f}")
        report_lines.append("")
        
        # Topic coverage
        if metrics.synthetic_topic_coverage and metrics.real_topic_coverage:
            report_lines.append("## Topic Coverage")
            report_lines.append(f"- Topic coverage similarity: {metrics.topic_coverage_similarity:.3f}")
            report_lines.append("")
        
        return "\n".join(report_lines)


def create_review_comparator(config: Dict[str, Any]) -> ReviewComparator:
    """Factory function to create a review comparator."""
    return ReviewComparator(config)