#!/usr/bin/env python3
"""
Command-line interface for the Synthetic Review Generator.
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from generator.review_generator import ReviewGenerator
from generator.regeneration import QualityControlledGenerator
from quality.comparison import ReviewComparator
from report.generate_report import ReportGenerator


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('synthetic_review_generator.log')
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)


def progress_callback(current: int, total: int, stats: Optional[dict] = None):
    """Progress callback for generation."""
    percentage = (current / total) * 100
    
    # Create visual progress bar
    bar_width = 40
    filled_width = int(bar_width * current / total)
    bar = '█' * filled_width + '░' * (bar_width - filled_width)
    
    print(f"\r🔄 Progress: [{bar}] {current}/{total} ({percentage:.1f}%)", end="", flush=True)
    
    if current == total:
        print()  # New line when complete
        if stats:
            print(f"✅ Quality stats - Pass rate: {stats.get('final_pass_rate', 0):.1%}, "
                  f"Regeneration rate: {stats.get('regeneration_rate', 0):.1%}")


def generate_command(args):
    """Execute the generate command."""
    print("🚀 Starting synthetic review generation...")
    
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        return 1
    
    try:
        # Initialize generator
        print("📋 Loading configuration...")
        generator = ReviewGenerator(config_path)
        
        # Validate configuration
        validation = generator.validate_configuration()
        if not validation["valid"]:
            print("❌ Configuration validation failed:")
            for error in validation["errors"]:
                print(f"   - {error}")
            return 1
        
        if validation["warnings"]:
            print("⚠️  Configuration warnings:")
            for warning in validation["warnings"]:
                print(f"   - {warning}")
        
        # Use quality-controlled generation if enabled
        use_quality_control = getattr(args, 'quality_control', True)
        
        if use_quality_control:
            print("🔍 Using quality-controlled generation...")
            quality_generator = QualityControlledGenerator(generator.config, generator)
            
            # Estimate time
            estimated_time = generator.estimate_generation_time(generator.target_reviews)
            print(f"⏱️  Estimated generation time: {estimated_time // 60}m {estimated_time % 60}s")
            
            # Generate dataset
            dataset = quality_generator.generate_quality_controlled_dataset(
                generator.target_reviews, 
                progress_callback
            )
        else:
            print("📝 Using basic generation...")
            dataset = generator.generate_dataset(progress_callback)
        
        # Save results
        output_file = generator.config["output"]["synthetic_reviews_file"]
        print(f"💾 Saving to {output_file}...")
        dataset.save_to_json(output_file)
        
        # Print summary
        synthetic_reviews = dataset.get_synthetic_reviews()
        print(f"✅ Generation completed!")
        print(f"   - Generated: {len(synthetic_reviews)} reviews")
        print(f"   - Average length: {sum(r.word_count for r in synthetic_reviews) / len(synthetic_reviews):.1f} words")
        
        # Print model distribution
        model_counts = {}
        for review in synthetic_reviews:
            if review.model_metadata:
                provider = review.model_metadata.provider
                model_counts[provider] = model_counts.get(provider, 0) + 1
        
        print(f"   - Model distribution:")
        for provider, count in model_counts.items():
            percentage = (count / len(synthetic_reviews)) * 100
            print(f"     • {provider}: {count} ({percentage:.1f}%)")
        
        # Quality stats if available
        if hasattr(dataset, 'metadata') and 'quality_stats' in dataset.metadata:
            quality_stats = dataset.metadata['quality_stats']
            print(f"   - Quality pass rate: {quality_stats.get('final_pass_rate', 0):.1%}")
            print(f"   - Regeneration rate: {quality_stats.get('regeneration_rate', 0):.1%}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Generation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        logging.exception("Generation failed")
        return 1


def report_command(args):
    """Execute the report command."""
    print("📊 Generating quality report...")
    
    try:
        # Load configuration
        config_path = args.config
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            return 1
        
        report_generator = ReportGenerator(config_path)
        
        # Load synthetic reviews
        synthetic_file = getattr(args, 'synthetic_file', None)
        if not synthetic_file:
            # Try to load from config
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            synthetic_file = config["output"]["synthetic_reviews_file"]
        
        if not os.path.exists(synthetic_file):
            print(f"❌ Synthetic reviews file not found: {synthetic_file}")
            print("   Run 'generate' command first to create synthetic reviews.")
            return 1
        
        # Load real reviews
        real_file = getattr(args, 'real_file', None)
        if not real_file:
            # Try to load from config
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            real_file = config["output"]["real_reviews_file"]
        
        print(f"📁 Loading synthetic reviews from {synthetic_file}...")
        print(f"📁 Loading real reviews from {real_file}...")
        
        # Generate report
        report_content = report_generator.generate_comprehensive_report(
            synthetic_file, 
            real_file
        )
        
        # Save report
        output_file = getattr(args, 'output', None)
        if not output_file:
            # Default from config
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            output_file = config["output"]["quality_report_file"]
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ Report generated: {output_file}")
        
        # Print summary to console
        lines = report_content.split('\n')
        summary_started = False
        for line in lines[:50]:  # First 50 lines for console summary
            if line.startswith('## Summary') or summary_started:
                summary_started = True
                print(line)
                if line.startswith('##') and not line.startswith('## Summary'):
                    break
        
        return 0
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        logging.exception("Report generation failed")
        return 1


def validate_command(args):
    """Execute the validate command."""
    print("🔍 Validating configuration...")
    
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        return 1
    
    try:
        generator = ReviewGenerator(config_path)
        validation = generator.validate_configuration()
        
        if validation["valid"]:
            print("✅ Configuration is valid!")
            
            # Print model provider status
            print("\n📡 Model Provider Status:")
            for name, provider in generator.model_providers.items():
                status = "✅ Available" if provider.is_available() else "❌ Not Available"
                print(f"   - {name}: {status}")
            
            # Print persona information
            personas = generator.persona_manager.get_all_personas()
            print(f"\n👤 Personas: {len(personas)} loaded")
            for persona in personas:
                print(f"   - {persona.role} ({persona.experience.value}, {persona.tone.value})")
            
            if validation["warnings"]:
                print("\n⚠️  Warnings:")
                for warning in validation["warnings"]:
                    print(f"   - {warning}")
        
        else:
            print("❌ Configuration validation failed:")
            for error in validation["errors"]:
                print(f"   - {error}")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        logging.exception("Validation failed")
        return 1


def info_command(args):
    """Execute the info command to show system information."""
    print("ℹ️  Synthetic Review Generator - System Information")
    print()
    
    # Python version
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    # Check dependencies
    dependencies = {
        "numpy": "Required for numerical computations",
        "scipy": "Optional for statistical analysis",
        "sklearn": "Optional for similarity computations", 
        "sentence_transformers": "Optional for semantic similarity",
        "requests": "Required for API calls",
        "yaml": "Required for configuration"
    }
    
    print("📦 Dependencies:")
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            status = "✅ Installed"
        except ImportError:
            status = "❌ Not installed"
        
        print(f"   - {dep}: {status} - {description}")
    
    print()
    
    # Check model provider availability
    try:
        config_path = getattr(args, 'config', 'config/generator.yaml')
        if os.path.exists(config_path):
            print("🤖 Model Provider Availability:")
            generator = ReviewGenerator(config_path)
            
            for name, provider in generator.model_providers.items():
                try:
                    available = provider.is_available()
                    status = "✅ Available" if available else "❌ Not available"
                    print(f"   - {name}: {status}")
                except Exception as e:
                    print(f"   - {name}: ❌ Error - {e}")
        
        else:
            print("⚠️  No configuration file found. Use --config to specify one.")
    
    except Exception as e:
        print(f"⚠️  Could not check model providers: {e}")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Synthetic Review Generator - Generate and analyze synthetic reviews",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s generate --config config/generator.yaml
  %(prog)s report --config config/generator.yaml
  %(prog)s validate --config config/generator.yaml
  %(prog)s info --config config/generator.yaml
        """
    )
    
    # Global arguments
    parser.add_argument(
        '--config', 
        default='config/generator.yaml',
        help='Path to configuration file (default: config/generator.yaml)'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set the logging level (default: INFO)'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate synthetic reviews')
    generate_parser.add_argument(
        '--no-quality-control',
        action='store_true',
        help='Disable quality control and regeneration (faster but lower quality)'
    )
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate quality and comparison report')
    report_parser.add_argument(
        '--synthetic-file',
        help='Path to synthetic reviews JSON file (default: from config)'
    )
    report_parser.add_argument(
        '--real-file', 
        help='Path to real reviews JSON file (default: from config)'
    )
    report_parser.add_argument(
        '--output',
        help='Output path for the report (default: from config)'
    )
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration and check system')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information and dependencies')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Handle commands
    if args.command == 'generate':
        # Invert the flag for internal use
        args.quality_control = not getattr(args, 'no_quality_control', False)
        return generate_command(args)
    elif args.command == 'report':
        return report_command(args)
    elif args.command == 'validate':
        return validate_command(args)
    elif args.command == 'info':
        return info_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())