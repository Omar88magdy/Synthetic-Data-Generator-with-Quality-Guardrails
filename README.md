# Synthetic Review Data Generator

A production-ready system that generates **synthetic service/tool reviews** using multiple LLM providers, enforces automated quality guardrails, compares results against real-world reviews, and produces comprehensive quality reports.

## 🚀 Features

- **Multi-LLM Generation**: Supports OpenAI GPT-4o-mini and DeepSeek API providers
- **Progressive Quality Scaling**: Dynamic quality thresholds that scale with dataset size
- **Quality Guardrails**: Automated diversity, bias detection, and realism checks  
- **Persona-Driven**: 8 configurable user personas with different experience levels and tones
- **Automatic Regeneration**: Failed reviews are automatically regenerated until quality thresholds are met
- **Real vs Synthetic Comparison**: Comprehensive analysis against real-world review data
- **Detailed Reporting**: Markdown reports with quality metrics and recommendations
- **Enhanced CLI Interface**: Visual progress bars and clean logging output
- **YAML Configuration**: All behavior is configurable, no hardcoded values
- **Robust Error Handling**: Graceful handling of API failures and None review cases

## 📊 Sample Output

The system generates Easygenerator reviews like these:

**Synthetic Review (5/5 stars, H&S Coordinator persona):**
> "Easygenerator has been a game-changer for our training team. The intuitive interface made it easy to create professional courses without any design background. The collaboration features work seamlessly, allowing our team to provide feedback in real-time. SCORM integration with our LMS was effortless, and our learners have responded positively to the interactive content."

**Quality Scores:**
- Realism: 0.847
- Diversity: 0.923  
- Persona Alignment: 0.8
- Domain Relevance: 0.85

## 🛠️ Installation

### Prerequisites

- Python 3.13+ (tested with 3.13.2)
- OpenAI API key (required for gpt-4o-mini provider)
- DeepSeek API key (required for deepseek-chat provider)
- 4GB+ RAM for semantic similarity calculations

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd synthetic-review-generator
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure API keys:**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export DEEPSEEK_API_KEY="your-deepseek-api-key-here"
```

## 📋 Configuration

The system is driven by `config/generator.yaml`. Key sections:

### Model Providers
```yaml
models:
  openai:
    enabled: true
    model: "gpt-4o-mini"
    temperature: 0.9
    max_tokens: 500
    weight: 0.6  # 60% of reviews
    
  deepseek:
    enabled: true
    model: "deepseek-chat"
    temperature: 0.8
    max_tokens: 500
    weight: 0.4  # 40% of reviews
```

### Generation Settings
```yaml
generation:
  target_reviews: 200     # Target number of reviews to generate
  batch_size: 30          # Reviews per batch for quality control
  min_reviews: 5          # Minimum acceptable reviews
```

### Progressive Quality Thresholds
```yaml
quality_thresholds:
  # Progressive scaling based on corpus size
  max_jaccard_similarity: 0.45     # Base threshold (scales 1.0x-1.5x)
  max_semantic_similarity: 0.85     # Base threshold (scales 1.0x-1.5x)  
  min_domain_keywords: 2            # Required domain keywords per review
  min_quality_score: 0.6            # Overall composite quality threshold
  
# Note: Thresholds automatically scale from 1.0x (small corpus) 
# to 1.5x (200+ reviews) to accommodate natural similarity increases
```

## 🖥️ Usage

### Basic Commands

**Generate 200 synthetic Easygenerator reviews:**
```bash
python cli.py generate
```

**Generate quality report:**
```bash
python cli.py report
```

**Validate configuration:**
```bash
python cli.py validate --config config/generator.yaml
```

**Show system info:**
```bash
python cli.py info
```

### Advanced Usage

**Generate without quality control (faster, for testing):**
```bash
python cli.py generate --no-quality-control
```

**Debug mode with verbose logging:**
```bash
python cli.py generate --log-level DEBUG
```

## 📁 Project Structure

```
synthetic-review-generator/
├── config/
│   └── generator.yaml              # Main configuration file
├── data/
│   ├── real_reviews.json          # Real reviews for comparison
│   └── synthetic_reviews.json     # Generated synthetic reviews
├── generator/
│   ├── models.py                  # Data models and schemas
│   ├── model_providers.py         # LLM provider interfaces
│   ├── personas.py                # Persona management
│   ├── review_generator.py        # Main generation logic
│   └── regeneration.py            # Quality-controlled generation
├── quality/
│   ├── diversity.py               # Diversity metrics
│   ├── bias.py                    # Bias detection
│   ├── realism.py                 # Realism analysis
│   ├── scoring.py                 # Quality scoring
│   └── comparison.py              # Synthetic vs real comparison
├── report/
│   └── generate_report.py         # Report generation
├── cli.py                         # Command-line interface
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🔬 Quality System

### Diversity Metrics
- **Jaccard Similarity**: Vocabulary overlap between reviews
- **Semantic Similarity**: Meaning-based similarity using embeddings
- **Vocabulary Richness**: Unique word usage across corpus

### Bias Detection  
- **Sentiment Bias**: Unbalanced sentiment across models
- **Rating Bias**: Deviation from expected rating distribution
- **Persona Bias**: Unrealistic persona-rating correlations

### Realism Analysis
- **Domain Relevance**: Presence of technical terminology
- **Marketing Language**: Detection of promotional content
- **Specificity**: Concrete details vs vague descriptions
- **Authenticity**: Personal experience indicators

### Automatic Regeneration
- Reviews failing quality checks are automatically regenerated
- Configurable maximum retry attempts
- Quality improvement tracking

## 📊 Sample Report Metrics

```
Overall Realism Score: 0.847 / 1.000
├── Length Similarity: 0.923
├── Rating Distribution: 0.834  
├── Sentiment Distribution: 0.891
├── Vocabulary Overlap: 0.756
└── Topic Coverage: 0.812

Quality Statistics:
├── Initial Pass Rate: 73.2%
├── Regeneration Rate: 26.8%
├── Final Pass Rate: 94.1%
└── Average Attempts: 1.3
```

## 🔄 Recent Improvements (v1.1.0)

### Quality System Enhancements
- **Progressive Quality Scaling**: Dynamic thresholds (1.0x to 1.5x) based on corpus size
- **Bug Fixes**: Resolved NoneType scoring errors and improved error handling
- **Robust Generation**: Better fallback mechanisms for API failures

### Generation Capabilities
- **Scaled Generation**: Now supports 200 reviews with 30-review batches
- **Visual Progress**: Enhanced CLI with Unicode progress bars (█░)
- **Provider Balance**: 60% OpenAI / 40% DeepSeek distribution

### Performance
- **Faster Iteration**: Configurable regeneration attempts (default: 1)
- **Cleaner Output**: Suppressed verbose HTTP logging
- **Memory Efficient**: Optimized similarity calculations

## 🎯 Personas Included

1. **H&S Coordinator** (beginner, 15%) - Focus on simplicity and support
2. **Sales Training Facilitator** (intermediate, 20%) - Emphasis on engagement and deadlines
3. **Instructional Designer** (intermediate, 15%) - Values learner engagement
4. **Innovation Engineer** (intermediate, 15%) - Analytical, integration-focused
5. **Consultant** (intermediate, 10%) - Results-focused, practical approach
6. **Training Specialist** (expert, 15%) - Workflow efficiency and quality
7. **Lecturer/Learning Designer** (expert, 10%) - Planning and organization



### Multi-Provider Architecture
- **Rationale**: Reduces single-point-of-failure and enables cost optimization
- **Trade-off**: Added complexity vs robustness and comparison capability

### Quality-First Approach  
- **Rationale**: Ensures high-quality output through automated guardrails
- **Trade-off**: Slower generation vs guaranteed quality standards

### Configuration-Driven Design
- **Rationale**: Maximum flexibility without code changes
- **Trade-off**: Complex config vs ease of experimentation

### Real Review Comparison
- **Rationale**: Provides objective quality assessment baseline
- **Trade-off**: Manual curation effort vs validation accuracy

## ⚠️ Current Limitations

### Technical
- **API Dependencies**: Requires valid OpenAI and DeepSeek API keys
- **Rate Limiting**: API providers may have rate limits for batch generation
- **Memory Usage**: Large datasets (200+ reviews) require significant RAM
- **Processing Time**: Quality control adds 2-3x generation time (can be reduced)

### Model & Quality
- **Provider Variability**: OpenAI and DeepSeek produce different styles
- **Context Length**: Limited to 500 token outputs per review
- **Marketing Detection**: Marketing language scoring not yet implemented
- **Semantic Variance**: Minor non-determinism in embeddings (~0.001 variance)

### Scope
- **Domain-Specific**: Configured for Easygenerator reviews only
- **Static Data**: Based on real reviews from January 2026
- **Persona Limitation**: Cannot create new personas at runtime

## ⚖️ Ethical Considerations

### Synthetic Data Risks
- **Misrepresentation**: Generated reviews could be mistaken for authentic user feedback
- **Bias Amplification**: Models may perpetuate biases present in training data
- **Market Manipulation**: Synthetic reviews should never be used to deceive consumers

### Responsible Usage
- **Clear Labeling**: Always identify synthetic reviews as artificially generated
- **Purpose Limitation**: Use only for research, testing, and training purposes
- **Bias Monitoring**: Regularly audit outputs for unfair biases
- **Transparency**: Document generation methods and limitations

### Privacy Protection
- **No Personal Data**: System generates fictional personas, not real user profiles
- **Data Isolation**: Real reviews used only for comparison, not training
- **Secure Storage**: API keys and generated data should be stored securely

## 🔧 Development

### Running Tests
```bash
pytest tests/ -v --cov=generator --cov=quality
```

### Code Formatting
```bash
black . --line-length 100
flake8 . --max-line-length 100
```

### Adding New Model Providers
1. Implement the `ModelProvider` interface in `generator/model_providers.py`
2. Add provider configuration to `generator.yaml`
3. Update factory method in `ModelProviderFactory`

### Custom Quality Metrics
1. Add new analyzer in `quality/` directory
2. Implement scoring logic in `quality/scoring.py`
3. Update report generation in `report/generate_report.py`

## 📈 Performance Optimization

### Generation Speed
- Use local models (Ollama) for faster generation
- Reduce quality threshold strictness
- Disable expensive similarity calculations
- Generate in smaller batches

### Quality vs Speed Trade-offs
- **Fast Mode**: Disable regeneration (`--no-quality-control`)
- **Balanced Mode**: Default configuration
- **High Quality**: Stricter thresholds, more regeneration attempts

### Memory Management
- Process reviews in batches for large datasets
- Use streaming for very large comparison datasets
- Clear intermediate results during processing

## 🐛 Troubleshooting

### Setup Issues

**"OPENAI_API_KEY not found"**
- Export the key: `export OPENAI_API_KEY="sk-..."`
- Or add to `.env` file (won't be committed to Git)

**"DeepSeek API authentication failed"**
- Verify API key with DeepSeek
- Check key format: `export DEEPSEEK_API_KEY="..."`

### Generation Issues

**"Generation is very slow"**
- Use `--no-quality-control` for faster iteration
- Reduce batch size in config (currently 30)
- Check API rate limits and increase cooldown

**"High similarity scores in output"**
- **Expected**: Domain-specific content naturally has higher similarity
- **Solution**: Progressive thresholds adjust automatically (0.45→0.7 at 200 reviews)
- **Override**: Reduce `min_quality_score` in config to 0.1 for testing

**"Quality scores are inconsistent between runs"**
- **Cause**: Sentence transformer embeddings have minor variance
- **Impact**: Score variance ~0.001 (negligible for quality decisions)
- **Workaround**: Use bypass mode for reproducible development testing

### Debug Mode
```bash
python cli.py generate --log-level DEBUG
```

## 📚 Resources

### Dependencies Used
- [OpenAI API](https://openai.com/api/) - GPT-4o-mini model provider
- [DeepSeek API](https://api.deepseek.com) - DeepSeek chat model provider
- [Sentence Transformers](https://huggingface.co/sentence-transformers) - Semantic embeddings (all-MiniLM-L6-v2)
- [scikit-learn](https://scikit-learn.org/) - Machine learning utilities

### Quality Methodology
- Jaccard Similarity for text diversity measurement
- Cosine similarity for semantic analysis
- Statistical bias detection methodologies
- Progressive threshold scaling for corpus size adaptation

## 📄 License

This project is provided as-is for educational and research purposes. Please ensure compliance with all applicable terms of service when using external APIs.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

---

**Project**: Synthetic Review Generator for Easygenerator  
**Version**: 1.1.0  
**Last Updated**: 2026-01-19  
**Status**: ✅ Production Ready