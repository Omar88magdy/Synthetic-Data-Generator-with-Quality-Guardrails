# Data Analysis and Cleaning Scripts

This directory contains utility scripts for analyzing and cleaning the Easygenerator review dataset.

## Scripts Overview

### 1. `filter_null_job_titles.py`
**Purpose**: Remove reviews where job_title is null from the dataset.

**Usage**:
```bash
python scripts/filter_null_job_titles.py
```

**What it does**:
- Loads the review dataset
- Filters out entries with null job titles
- Updates the original file with cleaned data
- Reports the number of reviews removed

### 2. `analyze_reviews.py`
**Purpose**: Comprehensive analysis of review dataset for insights.

**Usage**:
```bash
python scripts/analyze_reviews.py
```

**Analysis includes**:
- **Domain Keywords**: Frequency analysis of e-learning terms
- **Job Title Categorization**: Groups reviewers by role type
- **Rating Distribution**: Shows star rating patterns
- **Review Length Analysis**: Word count statistics and distribution

**Output Example**:
```
=== DOMAIN KEYWORDS FROM REAL REVIEWS ===
course: 38
training: 29
content: 31
learning: 22
...

=== JOB TITLE CATEGORIES ===
Learning & Development (12 total, 10 unique):
  - Learning And Development Specialist (2)
  - Director of Learning & Development (1)
  ...
```

### 3. `convert_review_format.py`
**Purpose**: Convert detailed G2 review format to simplified format for comparison.

**Usage**:
```bash
python scripts/convert_review_format.py
```

**What it does**:
- Converts complex G2 review structure to simple comparison format
- Combines likes/dislikes/benefits into single review text
- Determines sentiment based on rating
- Adds metadata while preserving job title information
- Outputs to `data/real_reviews.json`

**Output Format**:
```json
{
  "id": "easygenerator_real_review_001",
  "rating": 5,
  "review_text": "Combined review content...",
  "timestamp": "2025-11-27T12:00:00",
  "sentiment": "positive",
  "word_count": 150,
  "is_synthetic": false,
  "metadata": {
    "job_title": "Learning & Development Specialist",
    "company_size": "Mid-Market (51-1000 emp.)",
    "original_title": "Great authoring platform"
  }
}
```

## Usage Notes

- **Run from project root**: All scripts expect to be run from the project root directory
- **Backup data**: Scripts modify data files - consider backing up original data
- **Python 3**: All scripts require Python 3.6+
- **Dependencies**: Only standard library modules used (json, os, collections)

## Analysis Workflow

1. **Clean Data**: `python scripts/filter_null_job_titles.py`
2. **Analyze Dataset**: `python scripts/analyze_reviews.py`
3. **Convert Format**: `python scripts/convert_review_format.py`
4. **Update Config**: Use analysis results to update `config/generator.yaml`

## Integration with Main System

These scripts support the synthetic review generation by:
- **Cleaning real data** for accurate comparison
- **Extracting insights** for authentic persona and keyword configuration
- **Converting formats** for seamless integration with comparison logic
- **Providing statistics** for quality assessment baselines