#!/usr/bin/env python3
"""
Script to extract domain keywords using NER-like approach.
Usage: python scripts/analyze_reviews.py

This script extracts domain keywords by:
- Finding noun phrases and technical terms
- Identifying domain-specific entities
- Extracting meaningful concepts from context
"""

import json
import os
import re
from collections import Counter

def extract_domain_entities(text):
    """Extract domain-specific entities using NER-like approach."""
    
    # Convert to lowercase for processing
    text = text.lower()
    
    # Define patterns for domain entities
    patterns = {
        'tools_platforms': r'\b(easygenerator|articulate|captivate|storyline|rise|lectora|adobe|powerpoint|word)\b',
        'formats_standards': r'\b(scorm|xapi|tin can|aicc|html5|pdf|mp4|pptx)\b',
        'systems': r'\b(lms|learning management system|moodle|canvas|blackboard|cornerstone|workday|sharepoint)\b',
        'features': r'\b(drag[\-\s]?and[\-\s]?drop|templates?|quizzes?|assessments?|analytics|reporting|collaboration|branding|customization)\b',
        'content_types': r'\b(courses?|training|content|materials?|modules?|lessons?|videos?|interactive content|multimedia|scenarios?)\b',
        'user_roles': r'\b(instructional designer|training manager|learning specialist|developer|consultant|educator|trainer)\b',
        'processes': r'\b(authoring|course creation|content development|publishing|deployment|integration|onboarding)\b',
        'qualities': r'\b(user[\-\s]?friendly|intuitive|responsive|mobile[\-\s]?friendly|accessible|scalable)\b'
    }
    
    entities = []
    
    # Extract entities using patterns
    for category, pattern in patterns.items():
        matches = re.findall(pattern, text)
        for match in matches:
            # Clean up the match
            clean_match = re.sub(r'[\-\s]+', ' ', match).strip()
            if clean_match and len(clean_match) > 2:
                entities.append(clean_match)
    
    # Extract noun phrases (simple approach)
    # Look for adjective + noun combinations
    noun_phrases = re.findall(r'\b(?:easy|quick|fast|simple|advanced|professional|interactive|mobile|online|digital|custom)\s+(?:creation|development|design|interface|platform|tool|system|content|training|learning|course)\b', text)
    entities.extend(noun_phrases)
    
    # Extract compound terms
    compound_terms = re.findall(r'\b(?:course|content|training|learning|template|quiz|video|mobile|user|customer)\s+(?:creation|development|management|design|builder|editor|support|experience|interface|platform)\b', text)
    entities.extend(compound_terms)
    
    return entities

def analyze_domain_keywords_ner(reviews, min_frequency=2):
    """Extract domain keywords using NER-like approach."""
    
    print('=== DOMAIN ENTITY EXTRACTION (NER-like approach) ===')
    
    all_entities = []
    
    # Process each review
    for review in reviews:
        content = review['review']['content']
        
        # Combine all text content
        full_text = ' '.join([
            content.get('likes', ''),
            content.get('dislikes', ''),
            content.get('benefits', ''),
            review['review'].get('title', '')
        ])
        
        # Extract entities from this review
        entities = extract_domain_entities(full_text)
        all_entities.extend(entities)
    
    # Count entity frequencies
    entity_counts = Counter(all_entities)
    
    # Filter by minimum frequency and clean up
    filtered_entities = {}
    for entity, count in entity_counts.items():
        if count >= min_frequency and len(entity) > 2:
            # Normalize the entity
            normalized = entity.strip().lower()
            if normalized not in filtered_entities or count > filtered_entities[normalized][1]:
                filtered_entities[normalized] = (entity, count)
    
    # Sort by frequency
    sorted_entities = sorted(filtered_entities.values(), key=lambda x: x[1], reverse=True)
    
    print('Extracted domain entities (frequency >= {}):'.format(min_frequency))
    for entity, count in sorted_entities:
        print(f'  "{entity}": {count}')
    
    # Return as simple list for configuration
    domain_keywords = [entity for entity, count in sorted_entities if count >= min_frequency]
    
    print(f'\nGenerated {len(domain_keywords)} domain keywords')
    print('\nFor generator.yaml:')
    print('domain_keywords:')
    for keyword in domain_keywords[:20]:  # Top 20
        print(f'  - "{keyword}"')
    
    return domain_keywords

def categorize_job_titles_simple(reviews):
    """Simple job title categorization."""
    
    job_titles = [r['reviewer']['job_title'] for r in reviews if r['reviewer']['job_title']]
    unique_titles = sorted(set(job_titles))
    
    print(f'\n=== JOB TITLES ({len(unique_titles)} unique) ===')
    for title in unique_titles:
        count = job_titles.count(title)
        print(f'{title}: {count}')
    
    return unique_titles

def main():
    """Main analysis function - simplified."""
    
    input_file = 'data/easy_generator_reviews.json'
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        return
    
    # Load reviews
    with open(input_file, 'r') as f:
        reviews = json.load(f)
    
    print(f"Analyzing {len(reviews)} Easygenerator reviews...")
    print("=" * 60)
    
    # Extract domain keywords using NER-like approach
    keywords = analyze_domain_keywords_ner(reviews)
    
    # Simple job title analysis
    titles = categorize_job_titles_simple(reviews)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("Copy the domain_keywords above to config/generator.yaml")

if __name__ == "__main__":
    main()
