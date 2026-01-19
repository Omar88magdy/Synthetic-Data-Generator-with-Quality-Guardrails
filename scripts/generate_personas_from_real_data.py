import json
import google.generativeai as genai
from collections import Counter, defaultdict
import statistics
import os

def load_real_reviews(file_path):
    """Load and return real reviews from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_job_title_stats(reviews):
    """Extract job title statistics and group reviews by job title"""
    job_title_counter = Counter()
    reviews_by_job = defaultdict(list)
    
    for review in reviews:
        job_title = review['reviewer']['job_title']
        rating = review['review']['rating']
        
        job_title_counter[job_title] += 1
        reviews_by_job[job_title].append({
            'rating': rating,
            'likes': review['review']['content']['likes'],
            'dislikes': review['review']['content'].get('dislikes', ''),
            'benefits': review['review']['content'].get('benefits', '')
        })
    
    # Get top 10 job titles
    top_job_titles = job_title_counter.most_common(10)
    
    # Calculate average ratings
    job_stats = []
    for job_title, count in top_job_titles:
        ratings = [r['rating'] for r in reviews_by_job[job_title]]
        avg_rating = statistics.mean(ratings)
        job_stats.append({
            'job_title': job_title,
            'count': count,
            'avg_rating': avg_rating,
            'reviews': reviews_by_job[job_title]
        })
    
    return job_stats

def analyze_job_title_with_deepseek(job_stats, api_key):
    """Use DeepSeek to analyze characteristics for each job title"""
    from openai import OpenAI
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    
    personas = []
    
    for job_data in job_stats:
        job_title = job_data['job_title']
        reviews = job_data['reviews']
        count = job_data['count']
        avg_rating = job_data['avg_rating']
        
        # Combine all review content for this job title
        all_likes = ' '.join([r['likes'] for r in reviews if r['likes']])
        all_dislikes = ' '.join([r['dislikes'] for r in reviews if r['dislikes']])
        all_benefits = ' '.join([r['benefits'] for r in reviews if r['benefits']])
        
        # Create prompt for DeepSeek
        prompt = f"""
        Analyze the following reviews from users with job title "{job_title}" ({count} reviews, {avg_rating:.2f} avg rating).
        
        LIKES: {all_likes}
        DISLIKES: {all_dislikes}
        BENEFITS: {all_benefits}
        
        Based on this analysis, generate persona characteristics in this exact format:
        - role: "{job_title}"
        - experience: [beginner/intermediate/expert based on content complexity]
        - tone: [professional/business-focused/results-focused/analytical/enthusiastic]
        - characteristics: (list 4-6 bullet points describing their focus areas, language patterns, and concerns)
        - weight: [suggest a weight from 0.10 to 0.30 based on sample size]
        
        Focus on:
        1. What features/aspects they emphasize most
        2. Technical vs non-technical language they use
        3. Common words/phrases they use
        4. Main concerns or dislikes they mention
        5. Their overall satisfaction level
        """
        
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=1.0,
                max_tokens=1000
            )
            personas.append({
                'job_title': job_title,
                'analysis': response.choices[0].message.content,
                'stats': {
                    'count': count,
                    'avg_rating': avg_rating
                }
            })
            print(f"✓ Analyzed {job_title}")
        except Exception as e:
            print(f"✗ Error analyzing {job_title}: {e}")
    
    return personas

def save_personas_to_json(personas, output_file):
    """Save persona analysis to JSON file and display results"""
    personas_data = []
    
    for persona in personas:
        print(f"\n=== {persona['job_title']} ===")
        print(f"Count: {persona['stats']['count']}, Avg Rating: {persona['stats']['avg_rating']:.2f}")
        print(f"DeepSeek Analysis:\n{persona['analysis']}")
        print("-" * 50)
        
        # Create structured data for JSON
        persona_data = {
            'job_title': persona['job_title'],
            'sample_count': persona['stats']['count'],
            'avg_rating': round(persona['stats']['avg_rating'], 2),
            'deepseek_analysis': persona['analysis']
        }
        personas_data.append(persona_data)
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(personas_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Personas saved to JSON: {output_file}")
    print(f"✓ Ready to copy and integrate into your generator.yaml")
    return personas_data

def main():
    # Configuration
    reviews_file = "/Users/omar/Easygenerator-assignment/synthetic-review-generator/data/easy_generator_real_reviews.json"
    deepseek_api_key = "your_deepseek_api_key_here"  # Update with your DeepSeek API key
    
    print("Loading real reviews...")
    reviews = load_real_reviews(reviews_file)
    
    print("Extracting job title statistics...")
    job_stats = extract_job_title_stats(reviews)
    
    print("\nTop 10 Job Titles:")
    for job_data in job_stats:
        print(f"- {job_data['job_title']}: {job_data['count']} reviews, {job_data['avg_rating']:.2f} avg rating")
    
    print("\nAnalyzing with DeepSeek...")
    personas = analyze_job_title_with_deepseek(job_stats, deepseek_api_key)
    
    print("\nSaving results to JSON...")
    output_file = "/Users/omar/Easygenerator-assignment/synthetic-review-generator/config/extracted_personas.json"
    save_personas_to_json(personas, output_file)

if __name__ == "__main__":
    main()
