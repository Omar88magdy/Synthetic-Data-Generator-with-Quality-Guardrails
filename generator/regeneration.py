"""
Regeneration integration module that ties together quality scoring and regeneration.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from generator.models import Review, ReviewDataset
from quality.scoring import QualityScorer
from .scoring import RegenerationManager


class QualityControlledGenerator:
    """Generator that automatically applies quality controls and regeneration."""
    
    def __init__(self, config: Dict[str, Any], review_generator):
        self.config = config
        self.review_generator = review_generator
        self.quality_scorer = QualityScorer(config)
        quality_threshold = config.get('quality_threshold', 0.6)
        self.regeneration_manager = RegenerationManager(quality_threshold)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Quality control statistics
        self.quality_stats = {
            "total_generated": 0,
            "passed_initial_quality": 0,
            "required_regeneration": 0,
            "final_quality_passes": 0,
            "quality_failures": 0
        }
    
    def generate_quality_controlled_review(self, existing_reviews: List[Review]) -> Optional[Review]:
        """Generate a single review with automatic quality control and regeneration."""
        self.quality_stats["total_generated"] += 1
        
        try:
            # Generate initial review
            provider_weights = {name: config.get("weight", 1.0) 
                              for name, config in self.config["models"].items() 
                              if config.get("enabled", False)}
            total_weight = sum(provider_weights.values())
            provider_distribution = {name: weight / total_weight 
                                   for name, weight in provider_weights.items()}
            
            initial_review = self.review_generator._generate_single_review(provider_distribution)
            
            if not initial_review:
                self.logger.warning("Failed to generate initial review")
                return None
            
            # Apply quality scoring
            quality_scores = self.quality_scorer.score_review(initial_review, existing_reviews)
            initial_review.quality_scores = quality_scores
            
            # Check if regeneration is needed
            passes, failures = self.quality_scorer.passes_quality_threshold(quality_scores, existing_reviews)
            
            if passes:
                self.quality_stats["passed_initial_quality"] += 1
                self.quality_stats["final_quality_passes"] += 1
                self.logger.debug(f"Review passed initial quality check: {quality_scores.composite_score:.3f}")
                return initial_review
            
            # Implement proper regeneration logic
            self.quality_stats["required_regeneration"] += 1
            self.logger.info(f"Review requires regeneration: {failures}")
            
            max_regeneration_attempts = self.config.get("regeneration", {}).get("max_attempts", 3)
            
            for attempt in range(max_regeneration_attempts):
                self.logger.info(f"Regeneration attempt {attempt + 1}/{max_regeneration_attempts}")
                
                # Generate new review
                regenerated_review = self.review_generator._generate_single_review(provider_distribution)
                if not regenerated_review:
                    self.logger.warning(f"Failed to generate review on regeneration attempt {attempt + 1}")
                    continue
                
                # Score the regenerated review
                regenerated_quality_scores = self.quality_scorer.score_review(regenerated_review, existing_reviews)
                regenerated_review.quality_scores = regenerated_quality_scores
                
                # Check if regenerated review passes
                regenerated_passes, regenerated_failures = self.quality_scorer.passes_quality_threshold(regenerated_quality_scores, existing_reviews)
                
                if regenerated_passes:
                    self.quality_stats["final_quality_passes"] += 1
                    self.logger.info(f"Regenerated review passed quality check: {regenerated_quality_scores.composite_score:.3f}")
                    return regenerated_review
                else:
                    self.logger.info(f"Regenerated review still failed: {regenerated_failures}")
            
            # If all regeneration attempts failed, count as failure and return best attempt
            self.quality_stats["quality_failures"] += 1
            self.logger.warning(f"All regeneration attempts failed, returning best attempt")
            return initial_review if initial_review else None
            
        except Exception as e:
            self.logger.error(f"Error in quality-controlled generation: {e}")
            return None
    
    def generate_quality_controlled_dataset(self, target_count: int, 
                                          progress_callback: Optional[callable] = None) -> ReviewDataset:
        """Generate a dataset with batch-based quality control and regeneration."""
        self.logger.info(f"Starting batch-based quality-controlled generation of {target_count} reviews")
        
        dataset = ReviewDataset()
        existing_reviews = []
        generated_count = 0
        batch_size = self.config["generation"].get("batch_size", 10)
        
        while generated_count < target_count:
            # Determine current batch size
            current_batch_size = min(batch_size, target_count - generated_count)
            self.logger.info(f"Generating batch {generated_count//batch_size + 1}: {current_batch_size} reviews")
            
            # Generate initial batch
            batch_reviews = self._generate_initial_batch(current_batch_size, existing_reviews)
            
            # Analyze batch quality and regenerate problematic reviews
            final_batch = self._process_batch_with_quality_control(batch_reviews, existing_reviews)
            
            # Add successful reviews to dataset
            for review in final_batch:
                if review:
                    dataset.add_review(review)
                    existing_reviews.append(review)
                    generated_count += 1
                    
                    if progress_callback:
                        progress_callback(generated_count, target_count, self.get_quality_stats())
            
            self.logger.info(f"Completed batch: {generated_count}/{target_count} reviews")
        
        # Update dataset metadata
        dataset.metadata.update({
            "quality_controlled": True,
            "batch_based": True,
            "batch_size": batch_size,
            "quality_stats": self.get_quality_stats(),
            "generation_timestamp": datetime.now().isoformat()
        })
        
        self.logger.info(f"Completed batch-based quality-controlled generation: {generated_count} reviews")
        return dataset
    
    def _generate_initial_batch(self, batch_size: int, existing_reviews: List[Review]) -> List[Optional[Review]]:
        """Generate an initial batch of reviews."""
        batch_reviews = []
        
        for _ in range(batch_size):
            review = self.generate_quality_controlled_review(existing_reviews + batch_reviews)
            # Always append, even None values, to maintain batch size tracking
            batch_reviews.append(review)
            
        return batch_reviews
    
    def _process_batch_with_quality_control(self, batch_reviews: List[Optional[Review]], 
                                          existing_reviews: List[Review]) -> List[Optional[Review]]:
        """Process a batch with quality control and selective regeneration."""
        valid_batch_reviews = [r for r in batch_reviews if r is not None]
        
        if len(valid_batch_reviews) < 2:
            return batch_reviews  # Can't do batch analysis with < 2 reviews
        
        # Analyze batch-level diversity issues
        batch_texts = [review.review_text for review in valid_batch_reviews]
        
        try:
            # Check if the whole batch has diversity issues
            batch_diversity = self.quality_scorer.diversity_analyzer.analyze_batch_diversity(batch_texts)
            
            avg_jaccard = batch_diversity.get("avg_jaccard_similarity", 0.0)
            avg_semantic = batch_diversity.get("avg_semantic_similarity", 0.0)
            
            # Log batch analysis results
            self.logger.info(f"Batch analysis - Jaccard: {avg_jaccard:.3f}, Semantic: {avg_semantic:.3f}")
            
            # Define batch-level thresholds
            batch_jaccard_threshold = 0.4   # Higher than individual (0.3) since it's batch average
            batch_semantic_threshold = 0.9  # Higher than individual (0.85) since it's batch average
            
            regeneration_needed = (avg_jaccard > batch_jaccard_threshold or 
                                 avg_semantic > batch_semantic_threshold)
            
            if regeneration_needed:
                self.logger.info(f"Batch needs regeneration - Jaccard: {avg_jaccard:.3f}, Semantic: {avg_semantic:.3f}")
                
                # Identify worst performing reviews in the batch
                problem_indices = self._identify_problematic_reviews(valid_batch_reviews, batch_diversity)
                
                # Regenerate only the problematic reviews
                for idx in problem_indices:
                    self.logger.info(f"Regenerating review {idx+1} in batch")
                    new_review = self.generate_quality_controlled_review(existing_reviews)
                    if new_review:
                        batch_reviews[idx] = new_review
            else:
                self.logger.debug(f"Batch passed diversity check - Jaccard: {avg_jaccard:.3f}, Semantic: {avg_semantic:.3f}")
                
        except Exception as e:
            self.logger.error(f"Error in batch quality control: {e}")
        
        return batch_reviews
    
    def _identify_problematic_reviews(self, batch_reviews: List[Review], 
                                    batch_diversity: Dict[str, Any]) -> List[int]:
        """Identify which reviews in the batch are causing quality issues."""
        # Simple approach: regenerate reviews with highest similarities
        problem_indices = []
        
        # For now, regenerate the last 2-3 reviews in problematic batches
        # This is a simplified approach - could be made more sophisticated
        num_to_regenerate = min(3, len(batch_reviews) // 2)
        problem_indices = list(range(len(batch_reviews) - num_to_regenerate, len(batch_reviews)))
        
        return problem_indices
    
    def get_quality_stats(self) -> Dict[str, Any]:
        """Get quality control statistics."""
        stats = self.quality_stats.copy()
        
        if stats["total_generated"] > 0:
            stats["initial_pass_rate"] = stats["passed_initial_quality"] / stats["total_generated"]
            stats["regeneration_rate"] = stats["required_regeneration"] / stats["total_generated"]
            stats["final_pass_rate"] = stats["final_quality_passes"] / stats["total_generated"]
            stats["failure_rate"] = stats["quality_failures"] / stats["total_generated"]
        else:
            stats["initial_pass_rate"] = 0.0
            stats["regeneration_rate"] = 0.0
            stats["final_pass_rate"] = 0.0
            stats["failure_rate"] = 0.0
        
        return stats
    
    def analyze_quality_patterns(self, dataset: ReviewDataset) -> Dict[str, Any]:
        """Analyze quality patterns across the generated dataset."""
        reviews = dataset.get_synthetic_reviews()
        
        if not reviews:
            return {"status": "no_data"}
        
        # Quality score distributions
        quality_scores = []
        component_scores = {
            "jaccard_similarity": [],
            "semantic_similarity": [],
            "domain_relevance": [],
            "specificity_score": [],
            "marketing_language_score": [],
            "authenticity": []
        }
        
        for review in reviews:
            if review.quality_scores:
                quality_scores.append(review.quality_scores.composite_score)
                
                component_scores["jaccard_similarity"].append(review.quality_scores.jaccard_similarity)
                component_scores["semantic_similarity"].append(review.quality_scores.semantic_similarity)
                component_scores["domain_relevance"].append(review.quality_scores.domain_relevance)
                component_scores["specificity_score"].append(review.quality_scores.specificity_score)
                component_scores["marketing_language_score"].append(review.quality_scores.marketing_language_score)
        
        analysis = {
            "total_reviews": len(reviews),
            "reviews_with_scores": len(quality_scores),
            "quality_distribution": self._calculate_distribution(quality_scores),
            "component_distributions": {},
            "quality_by_model": self._analyze_quality_by_model(reviews),
            "quality_by_persona": self._analyze_quality_by_persona(reviews)
        }
        
        # Component score distributions
        for component, scores in component_scores.items():
            if scores:
                analysis["component_distributions"][component] = self._calculate_distribution(scores)
        
        return analysis
    
    def _calculate_distribution(self, values: List[float]) -> Dict[str, Any]:
        """Calculate distribution statistics for a list of values."""
        if not values:
            return {}
        
        import numpy as np
        
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "quartiles": {
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75))
            }
        }
    
    def _analyze_quality_by_model(self, reviews: List[Review]) -> Dict[str, Any]:
        """Analyze quality scores by model provider."""
        model_quality = {}
        
        for review in reviews:
            if review.model_metadata and review.quality_scores:
                provider = review.model_metadata.provider
                
                if provider not in model_quality:
                    model_quality[provider] = []
                
                model_quality[provider].append(review.quality_scores.composite_score)
        
        # Calculate statistics for each model
        model_stats = {}
        for provider, scores in model_quality.items():
            model_stats[provider] = self._calculate_distribution(scores)
        
        return model_stats
    
    def _analyze_quality_by_persona(self, reviews: List[Review]) -> Dict[str, Any]:
        """Analyze quality scores by persona."""
        persona_quality = {}
        
        for review in reviews:
            if review.persona and review.quality_scores:
                role = review.persona.role
                
                if role not in persona_quality:
                    persona_quality[role] = []
                
                persona_quality[role].append(review.quality_scores.composite_score)
        
        # Calculate statistics for each persona
        persona_stats = {}
        for role, scores in persona_quality.items():
            persona_stats[role] = self._calculate_distribution(scores)
        
        return persona_stats
    
    def generate_quality_report(self, dataset: ReviewDataset) -> str:
        """Generate a quality report for the dataset."""
        quality_analysis = self.analyze_quality_patterns(dataset)
        quality_stats = self.get_quality_stats()
        regeneration_stats = self.regeneration_manager.get_regeneration_stats()
        
        report_lines = []
        report_lines.append("# Quality Control Report")
        report_lines.append("")
        
        # Generation statistics
        report_lines.append("## Generation Statistics")
        report_lines.append(f"- Total generation attempts: {quality_stats['total_generated']}")
        report_lines.append(f"- Initial quality pass rate: {quality_stats['initial_pass_rate']:.1%}")
        report_lines.append(f"- Regeneration rate: {quality_stats['regeneration_rate']:.1%}")
        report_lines.append(f"- Final quality pass rate: {quality_stats['final_pass_rate']:.1%}")
        report_lines.append("")
        
        # Regeneration statistics
        if regeneration_stats["total_regenerations"] > 0:
            report_lines.append("## Regeneration Statistics")
            report_lines.append(f"- Total regenerations: {regeneration_stats['total_regenerations']}")
            report_lines.append(f"- Regeneration success rate: {regeneration_stats['success_rate']:.1%}")
            report_lines.append(f"- Average attempts per regeneration: {regeneration_stats['average_attempts']:.1f}")
            report_lines.append("")
        
        # Quality distributions
        if quality_analysis.get("quality_distribution"):
            dist = quality_analysis["quality_distribution"]
            report_lines.append("## Quality Score Distribution")
            report_lines.append(f"- Mean quality score: {dist['mean']:.3f}")
            report_lines.append(f"- Standard deviation: {dist['std']:.3f}")
            report_lines.append(f"- Range: {dist['min']:.3f} - {dist['max']:.3f}")
            report_lines.append("")
        
        return "\n".join(report_lines)


def create_quality_controlled_generator(config: Dict[str, Any], review_generator) -> QualityControlledGenerator:
    """Factory function to create a quality-controlled generator."""
    return QualityControlledGenerator(config, review_generator)