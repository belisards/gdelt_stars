#!/usr/bin/env python3
"""
GDELT Stars Pipeline - Master orchestration script
Runs all steps in order: fetch, embed, cluster, visualize
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def print_header(title):
    """Print a formatted header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def print_step(step_num, total_steps, description):
    """Print step information."""
    print(f"\n[STEP {step_num}/{total_steps}] {description}")
    print("-" * 70)


def run_step(module_name, step_description):
    """
    Run a pipeline step by importing and executing its main function.

    Args:
        module_name: Name of the module to import
        step_description: Description of the step for logging

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Running {module_name}...")
        module = __import__(module_name)
        module.main()
        logger.info(f"✓ {step_description} completed successfully")
        return True
    except Exception as e:
        logger.error(f"✗ {step_description} failed: {e}")
        return False


def main():
    """Main pipeline execution."""
    print_header("GDELT STARS PIPELINE")
    print("This pipeline will:")
    print("  1. Fetch GDELT data for Brazil")
    print("  2. Enrich with embeddings using sentence-transformers")
    print("  3. Perform clustering and keyword extraction")
    print("  4. Generate interactive star visualization")
    print()

    response = input("Continue? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print("Pipeline cancelled.")
        return

    steps = [
        ('fetch_gdelt', 'Data fetching'),
        ('enrich_embeddings', 'Embedding generation'),
        ('cluster_analysis', 'Clustering and keyword extraction'),
        ('visualize_stars', 'Visualization generation')
    ]

    total_steps = len(steps)

    for i, (module_name, description) in enumerate(steps, 1):
        print_step(i, total_steps, description)

        success = run_step(module_name, description)

        if not success:
            print()
            print("=" * 70)
            print("  PIPELINE FAILED")
            print("=" * 70)
            print(f"\nError at step {i}: {description}")
            print("Please check the error messages above and try again.")
            sys.exit(1)

    print()
    print_header("PIPELINE COMPLETED SUCCESSFULLY!")
    print("Output files generated:")
    print("  • data/gdelt_brazil_data.csv")
    print("  • data/gdelt_brazil_data_enriched.csv")
    print("  • data/gdelt_brazil_data_clustered.csv")
    print("  • docs/index.html")
    print()
    print("Open 'docs/index.html' in your browser to explore!")
    print()


if __name__ == "__main__":
    main()
