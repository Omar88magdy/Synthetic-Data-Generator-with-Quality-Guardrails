#!/usr/bin/env python3
"""
Script to convert raw Easygenerator reviews to the format expected by the synthetic review generator.
Usage: python scripts/convert_review_format.py

This script converts the detailed G2 review format to the simplified format used by the comparison system.
"""

import json
import os
from datetime import datetime

def convert_to_simple_format(input_file='data/easy_generator_reviews.json', output_file='data/real_reviews.json'):
    """Convert detailed G2 reviews to simple format for comparison."""
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        return
    
    # Load the detailed reviews
    with open(input_file, 'r') as f:
        detailed_reviews = json.load(f)
    
    # Convert to simple format
    simple_reviews = []
    
    for i, review in enumerate(detailed_reviews):
        content = review['review']['content']
        
        # Combine all text content
        review_text = ""
        if content.get('likes'):
            review_text += content['likes'] + " "
        if content.get('dislikes'):
            review_text += content['dislikes'] + " "
        if content.get('benefits'):
            review_text += content['benefits']
        
        review_text = review_text.strip()
        
        # Determine sentiment based on rating
        rating = review['review']['rating']
        if rating >= 4.6:
            sentiment = "positive"
        elif rating <= 2:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Create simple review format
        simple_review = {
            "id": f"easygenerator_real_review_{i+1:03d}",
            "rating": float(rating),
            "review_text": review_text,
            "timestamp": review['reviewer']['date'] + "T12:00:00",
            "sentiment": sentiment,
            "word_count": len(review_text.split()),
            "is_synthetic": False,
            "metadata": {
                "job_title": review['reviewer']['job_title'],
                "company_size": review['reviewer']['company_size'],
                "original_title": review['review']['title']
            }
        }
        
        simple_reviews.append(simple_review)
    
    # Save converted reviews
    with open(output_file, 'w') as f:
        json.dump(simple_reviews, f, indent=2)
    
    print(f"Converted {len(detailed_reviews)} reviews from {input_file} to {output_file}")
    print(f"Average review length: {sum(r['word_count'] for r in simple_reviews) / len(simple_reviews):.1f} words")
    
    # Show rating distribution
    from collections import Counter
    ratings = Counter(round(r['rating'], 1) for r in simple_reviews)
    print("\nRating distribution:")
    for rating in sorted(ratings.keys()):
        print(f"  {rating} stars: {ratings[rating]} reviews")

def main():
    """Main conversion function."""
    convert_to_simple_format()

if __name__ == "__main__":
    main()