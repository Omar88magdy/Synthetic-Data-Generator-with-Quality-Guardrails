#!/usr/bin/env python3
"""
Test script for semantic similarity detection in synthetic reviews.
"""

import sys
import os

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quality.diversity import DiversityAnalyzer


def test_semantic_similarity():
    """Test the semantic similarity functionality."""
    
    # Create analyzer with config for both vocabulary and semantic diversity
    config = {
        "batch_size": 4,
        "quality_thresholds": {
            "max_vocabulary_homogeneity": 0.7,
            "max_semantic_similarity": 0.85,
            "max_common_vocab_ratio": 0.5
        }
    }
    
    analyzer = DiversityAnalyzer(config)
    
    print("=" * 70)
    print("Testing Semantic Similarity Detection")
    print("=" * 70)
    
    # Check if semantic model is available
    if analyzer.sentence_model:
        print("✅ Sentence transformer model loaded successfully")
    else:
        print("⚠️  Sentence transformer not available - will use vocabulary fallback")
    
    print("\n1. Semantic Similarity Test Results:")
    print("-" * 40)
    
    # Run built-in semantic similarity test
    semantic_results = analyzer.test_batch_semantic_similarity()
    
    for test_name, stats in semantic_results.items():
        print(f"\n{test_name}:")
        if isinstance(stats, dict):
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                elif isinstance(value, list) and len(value) <= 5:
                    formatted_values = [f"{v:.3f}" for v in value]
                    print(f"  {key}: {formatted_values}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {stats}")
    
    print("\n2. Comprehensive Diversity Test:")
    print("-" * 35)
    
    # Test both vocabulary and semantic together
    comprehensive_results = analyzer.test_comprehensive_diversity()
    
    vocab_results = comprehensive_results["mixed_case_vocabulary"]
    semantic_results = comprehensive_results["mixed_case_semantic"]
    analysis = comprehensive_results["analysis"]
    
    print(f"Mixed case analysis:")
    print(f"  Vocabulary homogeneity: {analysis['vocab_homogeneity']:.3f}")
    print(f"  Semantic similarity: {analysis['semantic_similarity']:.3f}")
    print(f"  Interpretation: {analysis['interpretation']}")
    
    print("\n3. Individual Similarity Comparisons:")
    print("-" * 38)
    
    # Test individual semantic similarity
    text1 = "This e-learning platform is excellent for creating courses"
    text2 = "The educational software is great for developing lessons"
    text3 = "Customer support response time needs improvement"
    
    if hasattr(analyzer, 'calculate_semantic_similarity'):
        sem_sim_1_2 = analyzer.calculate_semantic_similarity(text1, text2)
        sem_sim_1_3 = analyzer.calculate_semantic_similarity(text1, text3)
        
        print(f"Similar meaning texts: {sem_sim_1_2:.3f}")
        print(f"Different meaning texts: {sem_sim_1_3:.3f}")
    else:
        print("Semantic similarity method not available")
    
    # Compare with Jaccard for reference
    jaccard_1_2 = analyzer.calculate_jaccard_similarity(text1, text2)
    jaccard_1_3 = analyzer.calculate_jaccard_similarity(text1, text3)
    
    print(f"\nFor comparison (Jaccard):")
    print(f"Similar meaning texts (Jaccard): {jaccard_1_2:.3f}")
    print(f"Different meaning texts (Jaccard): {jaccard_1_3:.3f}")
    
    print("\n" + "=" * 70)
    print("Semantic Similarity Interpretation:")
    print("- Higher semantic similarity (closer to 1.0) = MORE similar meaning = LESS diverse")
    print("- Semantic similarity captures meaning beyond just word overlap")
    print("- Useful for detecting paraphrases and conceptual similarity")
    print("- Complements vocabulary homogeneity for comprehensive diversity analysis")
    print("=" * 70)


if __name__ == "__main__":
    test_semantic_similarity()