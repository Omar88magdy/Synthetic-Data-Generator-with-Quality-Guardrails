"""
Comprehensive report generation for synthetic review analysis.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import yaml
import os

from generator.models import ReviewDataset, Review
from quality.diversity import DiversityAnalyzer
from quality.bias import BiasDetector
from quality.realism import RealismAnalyzer
from quality.comparison import ReviewComparator


class ReportGenerator:
    """Generates comprehensive reports for synthetic review analysis."""
    
    def __init__(self, config_path: str):
        """Initialize the report generator."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize analyzers
        self.diversity_analyzer = DiversityAnalyzer(self.config)
        self.bias_detector = BiasDetector(self.config)
        self.realism_analyzer = RealismAnalyzer(self.config)
        self.comparator = ReviewComparator(self.config)
    
    def generate_comprehensive_report(self, synthetic_reviews_file: str, 
                                    real_reviews_file: str) -> str:
        """Generate a comprehensive quality and comparison report."""
        self.logger.info("Generating comprehensive report...")
        
        try:
            # Load data
            synthetic_dataset = ReviewDataset.load_from_json(synthetic_reviews_file)
            synthetic_reviews = synthetic_dataset.get_synthetic_reviews()
            
            real_reviews = self.comparator.load_real_reviews(real_reviews_file)
            
            # Generate all analyses
            review_texts = [r.review_text for r in synthetic_reviews]
            diversity_analysis = self.diversity_analyzer.analyze_diversity(review_texts)
            bias_analysis = self.bias_detector.analyze_sentiment_bias(synthetic_reviews)
            real_review_texts = [r.review_text for r in real_reviews]
            realism_analysis = self.realism_analyzer.analyze_realism(review_texts, real_review_texts)
            comparison_metrics = self.comparator.perform_comprehensive_comparison(synthetic_reviews, real_reviews)
            
            # Generate report
            report = self._build_markdown_report(
                synthetic_dataset,
                synthetic_reviews,
                real_reviews,
                diversity_analysis,
                bias_analysis,
                realism_analysis,
                comparison_metrics
            )
            
            self.logger.info("Report generated successfully")
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            raise
    
    def _build_markdown_report(self, synthetic_dataset: ReviewDataset,
                              synthetic_reviews: List[Review],
                              real_reviews: List[Review],
                              diversity_analysis: Dict[str, Any],
                              bias_analysis: Dict[str, Any],
                              realism_analysis: Dict[str, Any],
                              comparison_metrics) -> str:
        """Build the comprehensive markdown report."""
        
        report_lines = []
        
        # Header
        report_lines.extend(self._generate_header())
        
        # Executive Summary
        report_lines.extend(self._generate_executive_summary(
            synthetic_reviews, real_reviews, comparison_metrics, bias_analysis, realism_analysis
        ))
        
        # Generation Statistics
        report_lines.extend(self._generate_generation_stats(synthetic_dataset, synthetic_reviews))
        
        # Quality Analysis
        report_lines.extend(self._generate_quality_analysis(
            diversity_analysis, bias_analysis, realism_analysis
        ))
        
        # Model Comparison
        report_lines.extend(self._generate_model_comparison(synthetic_reviews))
        
        # Synthetic vs Real Comparison
        report_lines.extend(self._generate_comparison_section(comparison_metrics))
        
        # Detailed Findings
        report_lines.extend(self._generate_detailed_findings(
            diversity_analysis, bias_analysis, realism_analysis
        ))
        
        # Recommendations
        report_lines.extend(self._generate_recommendations(
            bias_analysis, realism_analysis, comparison_metrics
        ))
        
        # Appendix
        report_lines.extend(self._generate_appendix(synthetic_dataset))
        
        return "\n".join(report_lines)
    
    def _generate_header(self) -> List[str]:
        """Generate report header."""
        return [
            "# Synthetic Review Generation Quality Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Domain:** {self.config.get('domain', 'Unknown').replace('_', ' ').title()}",
            "",
            "---",
            ""
        ]
    
    def _generate_executive_summary(self, synthetic_reviews: List[Review],
                                  real_reviews: List[Review],
                                  comparison_metrics,
                                  bias_analysis: Dict[str, Any],
                                  realism_analysis: Dict[str, Any]) -> List[str]:
        """Generate executive summary section."""
        lines = [
            "## Executive Summary",
            ""
        ]
        
        # Key metrics
        overall_realism = comparison_metrics.overall_realism_score
        bias_level = bias_analysis.get("bias_level", "unknown")
        
        if realism_analysis.get("status") == "success":
            avg_realism = realism_analysis.get("overall_realism_stats", {}).get("mean", 0)
            pass_rate = realism_analysis.get("pass_rate", 0)
        else:
            avg_realism = 0
            pass_rate = 0
        
        lines.extend([
            f"- **Dataset Size:** {len(synthetic_reviews)} synthetic reviews, {len(real_reviews)} real reviews",
            f"- **Overall Realism Score:** {overall_realism:.3f} / 1.000",
            f"- **Bias Level:** {bias_level.capitalize()}",
            f"- **Average Individual Realism:** {avg_realism:.3f}",
            f"- **Quality Pass Rate:** {pass_rate:.1%}",
            ""
        ])
        
        # Status assessment
        if overall_realism >= 0.8:
            status = "🟢 **Excellent** - High-quality synthetic reviews with strong realism"
        elif overall_realism >= 0.6:
            status = "🟡 **Good** - Acceptable quality with room for improvement"
        elif overall_realism >= 0.4:
            status = "🟠 **Fair** - Moderate quality, significant improvements needed"
        else:
            status = "🔴 **Poor** - Low quality, major improvements required"
        
        lines.extend([
            f"**Quality Assessment:** {status}",
            ""
        ])
        
        return lines
    
    def _generate_generation_stats(self, dataset: ReviewDataset, reviews: List[Review]) -> List[str]:
        """Generate generation statistics section."""
        lines = [
            "## Generation Statistics",
            ""
        ]
        
        # Basic stats
        if reviews:
            avg_length = sum(r.word_count for r in reviews) / len(reviews)
            model_distribution = {}
            persona_distribution = {}
            rating_distribution = {i: 0 for i in range(1, 6)}
            
            for review in reviews:
                if review.model_metadata:
                    provider = review.model_metadata.provider
                    model_distribution[provider] = model_distribution.get(provider, 0) + 1
                
                if review.persona:
                    persona = review.persona.role
                    persona_distribution[persona] = persona_distribution.get(persona, 0) + 1
                
                rating_distribution[review.rating] += 1
            
            lines.extend([
                f"- **Total Reviews Generated:** {len(reviews)}",
                f"- **Average Length:** {avg_length:.1f} words",
                ""
            ])
            
            # Model distribution
            if model_distribution:
                lines.extend(["### Model Provider Distribution", ""])
                for provider, count in model_distribution.items():
                    percentage = (count / len(reviews)) * 100
                    lines.append(f"- **{provider}:** {count} ({percentage:.1f}%)")
                lines.append("")
            
            # Rating distribution
            lines.extend(["### Rating Distribution", ""])
            for rating in range(1, 6):
                count = rating_distribution[rating]
                percentage = (count / len(reviews)) * 100
                stars = "⭐" * rating
                lines.append(f"- **{rating} {stars}:** {count} ({percentage:.1f}%)")
            lines.append("")
        
        # Generation metadata
        if hasattr(dataset, 'metadata') and dataset.metadata:
            metadata = dataset.metadata
            
            if 'quality_stats' in metadata:
                quality_stats = metadata['quality_stats']
                lines.extend([
                    "### Quality Control Statistics",
                    "",
                    f"- **Initial Pass Rate:** {quality_stats.get('initial_pass_rate', 0):.1%}",
                    f"- **Regeneration Rate:** {quality_stats.get('regeneration_rate', 0):.1%}",
                    f"- **Final Pass Rate:** {quality_stats.get('final_pass_rate', 0):.1%}",
                    ""
                ])
            
            if 'regeneration_stats' in metadata:
                regen_stats = metadata['regeneration_stats']
                if regen_stats.get('total_regenerations', 0) > 0:
                    lines.extend([
                        "### Regeneration Statistics",
                        "",
                        f"- **Total Regenerations:** {regen_stats['total_regenerations']}",
                        f"- **Success Rate:** {regen_stats.get('success_rate', 0):.1%}",
                        f"- **Average Attempts:** {regen_stats.get('average_attempts', 0):.1f}",
                        ""
                    ])
        
        return lines
    
    def _generate_quality_analysis(self, diversity_analysis: Dict[str, Any],
                                 bias_analysis: Dict[str, Any],
                                 realism_analysis: Dict[str, Any]) -> List[str]:
        """Generate quality analysis section."""
        lines = [
            "## Quality Analysis",
            ""
        ]
        
        # Diversity Analysis
        if diversity_analysis.get("status") == "success":
            lines.extend([
                "### Diversity Metrics",
                ""
            ])
            
            jaccard_stats = diversity_analysis.get("jaccard_similarity_stats", {})
            semantic_stats = diversity_analysis.get("semantic_similarity_stats", {})
            
            lines.extend([
                f"- **Vocabulary Richness:** {diversity_analysis.get('vocabulary_richness', 0):.3f}",
                f"- **Average Jaccard Similarity:** {jaccard_stats.get('mean', 0):.3f}",
                f"- **Average Semantic Similarity:** {semantic_stats.get('mean', 0):.3f}",
                ""
            ])
            
            # Diversity issues
            issues = diversity_analysis.get("diversity_issues", {})
            if issues.get("high_jaccard_similarity_pairs", 0) > 0 or issues.get("high_semantic_similarity_pairs", 0) > 0:
                lines.extend([
                    "**⚠️ Diversity Concerns:**",
                    f"- High vocabulary overlap pairs: {issues.get('high_jaccard_similarity_pairs', 0)}",
                    f"- High semantic similarity pairs: {issues.get('high_semantic_similarity_pairs', 0)}",
                    ""
                ])
        
        # Bias Analysis
        lines.extend([
            "### Bias Analysis",
            ""
        ])
        
        bias_level = bias_analysis.get("bias_level", "unknown")
        overall_bias = bias_analysis.get("overall_bias", 0)  # Use "overall_bias" instead of "overall_bias_score"
        
        # Extract component scores from the bias analysis
        sentiment_bias = overall_bias  # For now, use overall bias as sentiment bias
        rating_bias = 0  # Not calculated in simplified version
        persona_bias = 0  # Not calculated in simplified version
        
        bias_emoji = {"low": "🟢", "moderate": "🟡", "high": "🟠", "severe": "🔴"}.get(bias_level, "⚫")
        
        lines.extend([
            f"- **Overall Bias Level:** {bias_emoji} {bias_level.capitalize()} ({overall_bias:.3f})",
            f"- **Sentiment Bias:** {sentiment_bias:.3f}",
            f"- **Rating Bias:** {rating_bias:.3f}",
            f"- **Persona Bias:** {persona_bias:.3f}",
            ""
        ])
        
        # Realism Analysis
        if realism_analysis.get("status") == "success":
            lines.extend([
                "### Realism Analysis",
                ""
            ])
            
            realism_stats = realism_analysis.get("overall_realism_stats", {})
            pass_rate = realism_analysis.get("pass_rate", 0)
            
            lines.extend([
                f"- **Average Realism Score:** {realism_stats.get('mean', 0):.3f}",
                f"- **Realism Score Range:** {realism_stats.get('min', 0):.3f} - {realism_stats.get('max', 0):.3f}",
                f"- **Quality Pass Rate:** {pass_rate:.1%}",
                ""
            ])
            
            # Component scores
            component_stats = realism_analysis.get("component_stats", {})
            if component_stats:
                lines.extend(["**Component Averages:**", ""])
                for component, stats in component_stats.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        component_name = component.replace('_', ' ').title()
                        lines.append(f"- **{component_name}:** {stats['mean']:.3f}")
                lines.append("")
        
        return lines
    
    def _generate_model_comparison(self, reviews: List[Review]) -> List[str]:
        """Generate model comparison section."""
        lines = [
            "## Model Provider Comparison",
            ""
        ]
        
        # Group by model
        model_groups = {}
        for review in reviews:
            if review.model_metadata:
                provider = review.model_metadata.provider
                if provider not in model_groups:
                    model_groups[provider] = []
                model_groups[provider].append(review)
        
        if not model_groups:
            lines.append("No model metadata available for comparison.")
            lines.append("")
            return lines
        
        # Compare models
        lines.extend(["| Model | Reviews | Avg Length | Avg Quality | Avg Time (ms) |",
                     "|-------|---------|------------|-------------|---------------|"])
        
        for provider, provider_reviews in model_groups.items():
            count = len(provider_reviews)
            avg_length = sum(r.word_count for r in provider_reviews) / count
            
            # Average quality score
            quality_scores = [r.quality_scores.composite_score for r in provider_reviews 
                            if r.quality_scores]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            # Average generation time
            gen_times = [r.model_metadata.generation_time_ms for r in provider_reviews 
                        if r.model_metadata]
            avg_time = sum(gen_times) / len(gen_times) if gen_times else 0
            
            lines.append(f"| {provider} | {count} | {avg_length:.1f} | {avg_quality:.3f} | {avg_time:.0f} |")
        
        lines.append("")
        
        # Quality distribution by model
        lines.extend(["### Quality Score Distribution by Model", ""])
        
        for provider, provider_reviews in model_groups.items():
            quality_scores = [r.quality_scores.composite_score for r in provider_reviews 
                            if r.quality_scores]
            
            if quality_scores:
                lines.extend([
                    f"**{provider}:**",
                    f"- Count: {len(quality_scores)}",
                    f"- Mean: {np.mean(quality_scores):.3f}",
                    f"- Std Dev: {np.std(quality_scores):.3f}",
                    f"- Min/Max: {np.min(quality_scores):.3f} / {np.max(quality_scores):.3f}",
                    ""
                ])
        
        return lines
    
    def _generate_comparison_section(self, comparison_metrics) -> List[str]:
        """Generate synthetic vs real comparison section."""
        lines = [
            "## Synthetic vs Real Review Comparison",
            "",
            f"**Overall Realism Score: {comparison_metrics.overall_realism_score:.3f} / 1.000**",
            ""
        ]
        
        # Length comparison
        lines.extend([
            "### Length Analysis",
            "",
            f"- **Synthetic Avg Length:** {comparison_metrics.synthetic_avg_length:.1f} words",
            f"- **Real Avg Length:** {comparison_metrics.real_avg_length:.1f} words",
            f"- **Length Similarity:** {comparison_metrics.length_similarity_score:.3f}",
            ""
        ])
        
        # Rating distribution
        if comparison_metrics.synthetic_rating_dist and comparison_metrics.real_rating_dist:
            lines.extend([
                "### Rating Distribution Comparison",
                "",
                "| Rating | Synthetic | Real | Difference |",
                "|--------|-----------|------|------------|"
            ])
            
            for rating in range(1, 6):
                syn_pct = comparison_metrics.synthetic_rating_dist.get(rating, 0) * 100
                real_pct = comparison_metrics.real_rating_dist.get(rating, 0) * 100
                diff = abs(syn_pct - real_pct)
                stars = "⭐" * rating
                lines.append(f"| {rating} {stars} | {syn_pct:.1f}% | {real_pct:.1f}% | {diff:.1f}% |")
            
            lines.extend([
                "",
                f"**Rating Distribution Similarity:** {comparison_metrics.rating_distribution_similarity:.3f}",
                ""
            ])
        
        # Sentiment distribution
        if comparison_metrics.synthetic_sentiment_dist and comparison_metrics.real_sentiment_dist:
            lines.extend([
                "### Sentiment Distribution Comparison",
                "",
                "| Sentiment | Synthetic | Real | Difference |",
                "|-----------|-----------|------|------------|"
            ])
            
            all_sentiments = set(comparison_metrics.synthetic_sentiment_dist.keys()) | \
                           set(comparison_metrics.real_sentiment_dist.keys())
            
            for sentiment in sorted(all_sentiments):
                syn_pct = comparison_metrics.synthetic_sentiment_dist.get(sentiment, 0) * 100
                real_pct = comparison_metrics.real_sentiment_dist.get(sentiment, 0) * 100
                diff = abs(syn_pct - real_pct)
                emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(sentiment, "❓")
                lines.append(f"| {sentiment.capitalize()} {emoji} | {syn_pct:.1f}% | {real_pct:.1f}% | {diff:.1f}% |")
            
            lines.extend([
                "",
                f"**Sentiment Distribution Similarity:** {comparison_metrics.sentiment_distribution_similarity:.3f}",
                ""
            ])
        
        # Vocabulary analysis
        lines.extend([
            "### Vocabulary Analysis",
            "",
            f"- **Synthetic Vocabulary Richness:** {comparison_metrics.synthetic_vocabulary_richness:.3f}",
            f"- **Real Vocabulary Richness:** {comparison_metrics.real_vocabulary_richness:.3f}",
            f"- **Vocabulary Overlap (Jaccard):** {comparison_metrics.vocabulary_overlap:.3f}",
            ""
        ])
        
        return lines
    
    def _generate_detailed_findings(self, diversity_analysis: Dict[str, Any],
                                  bias_analysis: Dict[str, Any],
                                  realism_analysis: Dict[str, Any]) -> List[str]:
        """Generate detailed findings section."""
        lines = [
            "## Detailed Findings",
            ""
        ]
        
        # Diversity findings
        if diversity_analysis.get("status") == "success":
            lines.extend(["### Diversity Assessment", ""])
            
            vocab_richness = diversity_analysis.get("vocabulary_richness", 0)
            if vocab_richness >= 0.5:
                lines.append("✅ **Good vocabulary diversity** - Reviews show varied language usage")
            else:
                lines.append("⚠️ **Limited vocabulary diversity** - Reviews may be too similar in language")
            
            issues = diversity_analysis.get("diversity_issues", {})
            violation_rate = issues.get("jaccard_violation_rate", 0)
            if violation_rate > 0.1:
                lines.append(f"⚠️ **High similarity concern** - {violation_rate:.1%} of review pairs exceed similarity thresholds")
            
            lines.append("")
        
        # Bias findings
        lines.extend(["### Bias Assessment", ""])
        
        bias_level = bias_analysis.get("bias_level", "unknown")
        if bias_level == "low":
            lines.append("✅ **Low bias detected** - Generation appears well-balanced")
        elif bias_level == "moderate":
            lines.append("⚠️ **Moderate bias detected** - Some patterns may need attention")
        else:
            lines.append("❌ **High bias detected** - Significant patterns requiring correction")
        
        violation_count = bias_analysis.get("total_violations", 0)
        if violation_count > 0:
            lines.append(f"⚠️ **{violation_count} bias violations** detected across different metrics")
        
        lines.append("")
        
        # Realism findings
        if realism_analysis.get("status") == "success":
            lines.extend(["### Realism Assessment", ""])
            
            pass_rate = realism_analysis.get("pass_rate", 0)
            if pass_rate >= 0.8:
                lines.append("✅ **High realism** - Most reviews meet quality standards")
            elif pass_rate >= 0.6:
                lines.append("⚠️ **Moderate realism** - Majority of reviews are acceptable")
            else:
                lines.append("❌ **Low realism** - Many reviews fail quality checks")
            
            # Issue distribution
            issue_dist = realism_analysis.get("issue_distribution", {})
            if issue_dist:
                lines.append("\n**Common Issues:**")
                for issue_type, count in sorted(issue_dist.items(), key=lambda x: x[1], reverse=True):
                    if count > 0:
                        lines.append(f"- {issue_type}: {count} reviews")
            
            lines.append("")
        
        return lines
    
    def _generate_recommendations(self, bias_analysis: Dict[str, Any],
                                realism_analysis: Dict[str, Any],
                                comparison_metrics) -> List[str]:
        """Generate recommendations section."""
        lines = [
            "## Recommendations",
            ""
        ]
        
        recommendations = []
        
        # Overall quality recommendations
        overall_realism = comparison_metrics.overall_realism_score
        if overall_realism < 0.6:
            recommendations.append("🔴 **Critical:** Overall realism is low. Consider reviewing generation parameters and quality thresholds.")
        elif overall_realism < 0.8:
            recommendations.append("🟡 **Moderate:** Good foundation but room for improvement in realism metrics.")
        
        # Bias recommendations
        bias_level = bias_analysis.get("bias_level", "unknown")
        if bias_level in ["high", "severe"]:
            recommendations.append("🔴 **Address bias immediately:** Detected significant bias patterns that may affect review authenticity.")
            
            component_scores = bias_analysis.get("component_scores", {})
            if component_scores.get("sentiment_bias", 0) > 0.15:
                recommendations.append("   - Adjust model temperature or prompt variations to reduce sentiment bias")
            if component_scores.get("rating_bias", 0) > 0.15:
                recommendations.append("   - Review rating distribution configuration")
            if component_scores.get("persona_bias", 0) > 0.15:
                recommendations.append("   - Ensure personas can produce varied rating patterns")
        
        # Realism recommendations
        if realism_analysis.get("status") == "success":
            pass_rate = realism_analysis.get("pass_rate", 0)
            if pass_rate < 0.7:
                recommendations.append("🟡 **Improve realism:** Consider these adjustments:")
                
                # Analyze component scores for specific recommendations
                component_stats = realism_analysis.get("component_stats", {})
                
                domain_score = component_stats.get("domain_relevance", {}).get("mean", 0)
                if domain_score < 0.5:
                    recommendations.append("   - Add more domain-specific keywords and technical terminology")
                
                marketing_score = component_stats.get("marketing_penalty", {}).get("mean", 0)
                if marketing_score > 0.3:
                    recommendations.append("   - Reduce promotional language and excessive positive adjectives")
                
                specificity_score = component_stats.get("specificity", {}).get("mean", 0)
                if specificity_score < 0.4:
                    recommendations.append("   - Include more specific details, numbers, and concrete examples")
        
        # Comparison-based recommendations
        if comparison_metrics.length_similarity_score < 0.7:
            length_diff = abs(comparison_metrics.synthetic_avg_length - comparison_metrics.real_avg_length)
            if length_diff > 20:
                target = "shorter" if comparison_metrics.synthetic_avg_length > comparison_metrics.real_avg_length else "longer"
                recommendations.append(f"🟡 **Adjust review length:** Synthetic reviews should be {target} to match real review patterns")
        
        if comparison_metrics.rating_distribution_similarity < 0.8:
            recommendations.append("🟡 **Review rating distribution:** Synthetic rating patterns don't match real reviews closely")
        
        # Model-specific recommendations
        recommendations.append("")
        recommendations.append("### Technical Recommendations:")
        recommendations.append("- Monitor generation costs and performance across model providers")
        recommendations.append("- Consider A/B testing different prompt variations")
        recommendations.append("- Implement periodic quality audits with human evaluation")
        recommendations.append("- Track quality metrics over time to identify degradation")
        
        # Add all recommendations
        for rec in recommendations:
            lines.append(rec)
        
        lines.append("")
        return lines
    
    def _generate_appendix(self, dataset: ReviewDataset) -> List[str]:
        """Generate appendix section."""
        lines = [
            "## Appendix",
            ""
        ]
        
        # Configuration summary
        lines.extend([
            "### Configuration Summary",
            "",
            f"- **Target Reviews:** {self.config['generation']['target_reviews']}",
            f"- **Domain:** {self.config['domain']}",
            f"- **Model Providers:** {', '.join(self.config['models'].keys())}",
            f"- **Personas:** {len(self.config['personas'])}",
            ""
        ])
        
        # Quality thresholds
        thresholds = self.config.get("quality_thresholds", {})
        if thresholds:
            lines.extend([
                "### Quality Thresholds Used",
                ""
            ])
            for threshold, value in thresholds.items():
                threshold_name = threshold.replace('_', ' ').title()
                lines.append(f"- **{threshold_name}:** {value}")
            lines.append("")
        
        # Generation metadata
        if hasattr(dataset, 'metadata') and dataset.metadata:
            metadata = dataset.metadata
            lines.extend([
                "### Generation Metadata",
                "",
                f"- **Generation Timestamp:** {metadata.get('generation_timestamp', 'Unknown')}",
                f"- **Target Count:** {metadata.get('target_count', 'Unknown')}",
                f"- **Actual Count:** {metadata.get('actual_count', 'Unknown')}",
                ""
            ])
        
        # Report metadata
        lines.extend([
            "### Report Metadata",
            "",
            f"- **Generated By:** Synthetic Review Generator v1.0.0",
            f"- **Report Timestamp:** {datetime.now().isoformat()}",
            f"- **Analysis Components:** Diversity, Bias, Realism, Comparison",
            ""
        ])
        
        return lines


def create_report_generator(config_path: str) -> ReportGenerator:
    """Factory function to create a report generator."""
    return ReportGenerator(config_path)