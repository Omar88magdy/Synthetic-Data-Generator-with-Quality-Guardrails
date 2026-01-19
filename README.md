# Synthetic Review Data Generator

A production-ready system that generates **synthetic service/tool reviews** using multiple LLM providers, enforces automated quality guardrails, compares results against real-world reviews, and produces comprehensive quality reports.

## 🚀 Features

- **Multi-LLM Generation**: Supports OpenAI and Ollama model providers
- **Quality Guardrails**: Automated diversity, bias detection, and realism checks  
- **Persona-Driven**: Configurable user personas with different experience levels and tones
- **Automatic Regeneration**: Failed reviews are automatically regenerated until quality thresholds are met
- **Real vs Synthetic Comparison**: Comprehensive analysis against real-world review data
- **Detailed Reporting**: Markdown reports with quality metrics and recommendations
- **CLI Interface**: Easy-to-use command-line interface
- **YAML Configuration**: All behavior is configurable, no hardcoded values

## 📊 Sample Output

The system generates reviews like these:

**Synthetic Review (4/5 stars, Senior Developer persona):**
> "We've been using this API gateway for 6 months and it handles our microservices traffic well. The rate limiting is configurable and integrates smoothly with our Kubernetes deployment. Documentation could be more comprehensive for advanced configurations, but the basic setup was straightforward. Performance has been solid with sub-100ms latency for most requests."

**Quality Scores:**
- Realism: 0.847
- Diversity: 0.923  
- Bias: Low
- Domain Relevance: 0.912

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (optional, for OpenAI provider)
- Ollama installed locally (optional, for local LLM provider)

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

4. **Configure API keys (optional):**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

5. **Set up Ollama (optional):**
```bash
# Install Ollama from https://ollama.ai
ollama pull llama2  # or your preferred model
```

## 📋 Configuration

The system is driven by `config/generator.yaml`. Key sections:

### Model Providers
```yaml
models:
  openai:
    enabled: true
    model: "gpt-3.5-turbo"
    temperature: 0.8
    weight: 0.6
    
  ollama:
    enabled: true
    model: "llama2" 
    weight: 0.4
```

### Personas
```yaml
personas:
  - role: "Senior Developer"
    experience: "expert"
    tone: "analytical"
    characteristics:
      - "Technical depth in reviews"
      - "Focus on performance and scalability"
    weight: 0.25
```

### Quality Thresholds
```yaml
quality_thresholds:
  max_jaccard_similarity: 0.3      # Vocabulary overlap limit
  max_semantic_similarity: 0.85    # Semantic similarity limit
  min_domain_keywords: 2           # Required domain keywords
  min_quality_score: 0.6           # Overall quality threshold
```

## 🖥️ Usage

### Basic Commands

**Generate synthetic reviews:**
```bash
python cli.py generate --config config/generator.yaml
```

**Generate quality report:**
```bash
python cli.py report --config config/generator.yaml
```

**Validate configuration:**
```bash
python cli.py validate --config config/generator.yaml
```

**Show system info:**
```bash
python cli.py info --config config/generator.yaml
```

### Advanced Usage

**Generate without quality control (faster):**
```bash
python cli.py generate --config config/generator.yaml --no-quality-control
```

**Custom report paths:**
```bash
python cli.py report --synthetic-file data/my_reviews.json --real-file data/real_reviews.json --output reports/custom_report.md
```

**Debug mode:**
```bash
python cli.py generate --config config/generator.yaml --log-level DEBUG
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

## 🎯 Design Decisions

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

## ⚠️ Limitations

### Technical Limitations
- **Dependency on External APIs**: OpenAI provider requires internet and API access
- **Local Model Performance**: Ollama performance varies significantly by hardware
- **Memory Usage**: Large datasets may require significant RAM for analysis
- **Processing Time**: Quality control adds 2-3x generation time

### Model Limitations
- **LLM Consistency**: Different models produce varying quality and style
- **Context Length**: Limited by model context windows (typically 2K-4K tokens)
- **Domain Knowledge**: Models may lack deep domain expertise
- **Temporal Awareness**: Generated reviews may not reflect current tool versions

### Quality Limitations
- **Semantic Analysis Accuracy**: Embeddings may not capture nuanced meaning differences
- **Bias Detection Scope**: Limited to statistical patterns, may miss subtle biases
- **Realism Thresholds**: Configured thresholds may not suit all domains
- **Human Evaluation Gap**: Automated metrics don't replace human judgment

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

### Common Issues

**"No model providers available"**
- Check API keys are set correctly
- Verify Ollama is running (for local models)
- Review provider configuration in YAML

**"Generation is very slow"**  
- Consider using local models instead of API-based
- Reduce quality thresholds
- Disable expensive similarity calculations

**"High bias detected in report"**
- Adjust temperature settings
- Review persona configurations
- Check rating distribution settings

**"Low realism scores"**
- Add more domain-specific keywords
- Adjust specificity thresholds
- Review persona characteristics

### Debug Mode
Enable debug logging to diagnose issues:
```bash
python cli.py generate --log-level DEBUG
```

## 📚 References

### Academic Background
- Jaccard Similarity for text diversity measurement
- Jensen-Shannon Divergence for distribution comparison
- Sentence transformers for semantic similarity analysis
- Statistical bias detection methodologies

### Related Tools
- [OpenAI API](https://openai.com/api/) - Commercial LLM provider
- [Ollama](https://ollama.ai/) - Local LLM deployment
- [Sentence Transformers](https://huggingface.co/sentence-transformers) - Semantic embeddings
- [scikit-learn](https://scikit-learn.org/) - Machine learning utilities

## 📄 License

This project is provided as-is for educational and research purposes. Please ensure compliance with all applicable terms of service when using external APIs and models.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Run quality checks (`black`, `flake8`, `pytest`)
5. Submit a pull request

---

**Generated by:** Synthetic Review Generator v1.0.0  
**Last Updated:** 2026-01-17  
**Documentation Status:** ✅ Complete