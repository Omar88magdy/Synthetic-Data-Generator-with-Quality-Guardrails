#!/usr/bin/env python3
"""
Test script for vocabulary homogeneity detection in synthetic reviews.
"""

import sys
import os

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quality.diversity import DiversityAnalyzer
from generator.models import Review
from datetime import datetime


def test_vocabulary_homogeneity():
    """Test the vocabulary homogeneity functionality."""
    
    # Create analyzer with config for vocabulary homogeneity
    config = {
        "batch_size": 4,
        "quality_thresholds": {
            "max_vocabulary_homogeneity": 0.7,
            "max_common_vocab_ratio": 0.5
        }
    }
    
    analyzer = DiversityAnalyzer(config)
    
    print("=" * 60)
    print("Testing Vocabulary Homogeneity Detection")
    print("=" * 60)
    
    # Run built-in test
    print("\n1. Built-in Test Results:")
    test_results = analyzer.test_batch_vocabulary_homogeneity()
    
    for test_name, stats in test_results.items():
        print(f"\n{test_name}:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            elif isinstance(value, list) and len(value) <= 5:
                formatted_values = [f"{v:.3f}" for v in value]
                print(f"  {key}: {formatted_values}")
            else:
                print(f"  {key}: {value}")
    
    # Test with Review objects
    print("\n2. Testing with Review objects:")
    
    # Create sample reviews with varying vocabulary overlap
    reviews = []
    review_texts = [
        "Easygenerator is excellent for course creation and SCORM export functionality.",
        "Great authoring tool for creating interactive courses and training materials.",
        "The platform helps with drag-and-drop content development and LMS integration.",
        "Customer support team is responsive when you need help with technical features.",
        "Templates and customization options make corporate branding straightforward.",
        "Video integration and multimedia support enhance learner engagement significantly.",
        "Reporting and analytics dashboard helps track student progress effectively.",
        "Mobile-responsive templates ensure content works across all devices seamlessly."
    ]
    
    for i, text in enumerate(review_texts):
        review = Review(
            id=f"test_review_{i}",
            rating=4,
            review_text=text,
            timestamp=datetime.now(),
            is_synthetic=True
        )
        reviews.append(review)
    
    # Test diversity check with new vocabulary homogeneity approach
    new_review = Review(
        id="new_test_review",
        rating=4,
        review_text="Great platform for creating courses with excellent templates.",
        timestamp=datetime.now(),
        is_synthetic=True
    )
    
    diversity_result = analyzer.check_diversity_against_corpus(new_review, reviews)
    
    print("Diversity Check Results:")
    print(f"  Passes check: {diversity_result['passes_diversity_check']}")
    print(f"  Vocabulary homogeneity: {diversity_result['vocabulary_homogeneity']:.3f}")
    print(f"  Batch vocabulary stats:")
    for key, value in diversity_result['batch_vocabulary_stats'].items():
        if isinstance(value, float):
            print(f"    {key}: {value:.3f}")
        else:
            print(f"    {key}: {value}")
    if diversity_result['violations']:
        print(f"  Violations: {diversity_result['violations']}")
    
    print("\n3. Testing High Homogeneity Case:")
    # Test with very similar reviews (high vocabulary overlap)
    similar_review_texts = [
        "Great tool for creating courses",
        "Great platform for creating courses", 
        "Great software for creating courses",
        "Great app for creating courses"
    ]
    
    high_homogeneity_result = analyzer.calculate_batch_vocabulary_homogeneity(similar_review_texts)
    print(f"High homogeneity example:")
    print(f"  Vocabulary homogeneity: {high_homogeneity_result['vocabulary_homogeneity']:.3f}")
    print(f"  Common vocab ratio: {high_homogeneity_result['common_vocab_ratio']:.3f}")
    
    print("\n4. Testing Low Homogeneity Case:")
    # Test with diverse vocabulary
    diverse_review_texts = [
        "Exceptional authoring platform with advanced SCORM capabilities",
        "Customer support response time requires significant improvement unfortunately", 
        "Pricing structure remains competitive compared to industry standards",
        "Mobile-responsive templates enhance learner engagement dramatically"
    ]
    
    low_homogeneity_result = analyzer.calculate_batch_vocabulary_homogeneity(diverse_review_texts)
    print(f"Low homogeneity example:")
    print(f"  Vocabulary homogeneity: {low_homogeneity_result['vocabulary_homogeneity']:.3f}")
    print(f"  Common vocab ratio: {low_homogeneity_result['common_vocab_ratio']:.3f}")
    
    print("\n" + "=" * 60)
    print("Vocabulary Homogeneity Interpretation:")
    print("- Higher homogeneity values (closer to 1.0) = MORE homogeneous = LESS diverse")
    print("- Common vocab ratio shows proportion of shared vocabulary")  
    print("- Values above thresholds indicate insufficient vocabulary diversity")
    print("=" * 60)


if __name__ == "__main__":
    test_vocabulary_homogeneity()