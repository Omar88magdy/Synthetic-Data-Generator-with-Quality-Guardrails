#!/usr/bin/env python3
"""
Script to remove reviews with null job_title from the dataset.
Usage: python scripts/filter_null_job_titles.py
"""

import json
import os

def filter_null_job_titles(input_file='data/easy_generator_reviews.json'):
    """Remove reviews where job_title is null."""
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        return
    
    # Load the JSON file
    with open(input_file, 'r') as f:
        reviews = json.load(f)
    
    # Filter out reviews where job_title is null
    filtered_reviews = [
        review for review in reviews 
        if review.get('reviewer', {}).get('job_title') is not None
    ]
    
    print(f"Original count: {len(reviews)}")
    print(f"Filtered count: {len(filtered_reviews)}")
    print(f"Removed {len(reviews) - len(filtered_reviews)} reviews with null job_title")
    
    # Save the filtered reviews back to the file
    with open(input_file, 'w') as f:
        json.dump(filtered_reviews, f, indent=2)
    
    print(f"File {input_file} updated successfully!")

if __name__ == "__main__":
    filter_null_job_titles()