#!/usr/bin/env python3
"""
GDELT Embedding Enricher - Adds sentence embeddings to URL titles
Reads CSV file and adds embeddings using sentence-transformers.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_data(input_file='data/gdelt_brazil_data.csv'):
    """Load GDELT data from CSV file."""
    input_path = Path(__file__).parent.parent / input_file

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Loading data from: {input_path.absolute()}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} records")

    return df


def generate_embeddings(df, model_name='all-MiniLM-L6-v2', batch_size=32):
    """
    Generate embeddings for URL titles using sentence-transformers.

    Args:
        df: DataFrame with 'url_title' column
        model_name: Name of the sentence-transformers model to use
        batch_size: Batch size for encoding

    Returns:
        DataFrame with added embedding columns
    """
    if df.empty:
        logger.warning("Empty DataFrame provided")
        return df

    if 'url_title' not in df.columns:
        raise ValueError("DataFrame must contain 'url_title' column")

    df = df.copy()

    logger.info(f"Loading model: {model_name}")
    logger.info("Downloading model from HuggingFace (first run only)...")

    try:
        model = SentenceTransformer(model_name, trust_remote_code=True, token=False)
    except Exception as e:
        error_msg = str(e)
        if 'expired' in error_msg.lower() or '401' in error_msg:
            logger.error("HuggingFace token expired or authentication issue detected")
            logger.info("Attempting to load model without authentication...")
            try:
                import os
                os.environ.pop('HF_TOKEN', None)
                os.environ.pop('HUGGING_FACE_HUB_TOKEN', None)
                model = SentenceTransformer(model_name, use_auth_token=False, token=False)
            except Exception as e3:
                raise RuntimeError(
                    f"Could not load model '{model_name}'. "
                    "HuggingFace authentication token appears to be expired.\n"
                    "Please run: huggingface-cli logout\n"
                    "Then run: huggingface-cli login --token YOUR_NEW_TOKEN\n"
                    "Or simply try: rm -rf ~/.cache/huggingface/token"
                ) from e3
        else:
            raise RuntimeError(
                f"Could not load model '{model_name}': {error_msg}\n"
                "Please check your internet connection."
            ) from e

    titles_to_embed = df['url_title'].fillna('').astype(str).tolist()
    valid_titles = [t for t in titles_to_embed if t.strip()]

    logger.info(f"Generating embeddings for {len(valid_titles)} titles (batch_size={batch_size})...")

    embeddings = model.encode(
        titles_to_embed,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    logger.info(f"Generated embeddings with shape: {embeddings.shape}")

    for i in range(embeddings.shape[1]):
        df[f'embedding_{i}'] = embeddings[:, i]

    logger.info(f"Added {embeddings.shape[1]} embedding columns to DataFrame")

    logger.info("Reducing embeddings to 2D using t-SNE...")
    reducer = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, len(embeddings) - 1),
        max_iter=1000
    )
    coords_2d = reducer.fit_transform(embeddings)

    scaler = MinMaxScaler(feature_range=(0, 1))
    coords_2d = scaler.fit_transform(coords_2d)

    df['x_2d'] = coords_2d[:, 0]
    df['y_2d'] = coords_2d[:, 1]

    logger.info(f"Added 2D coordinates (x_2d, y_2d) to DataFrame")

    return df


def save_enriched_data(df, output_file='data/gdelt_brazil_data_enriched.csv'):
    """Save enriched DataFrame to CSV file."""
    output_path = Path(__file__).parent.parent / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    logger.info(f"Enriched data saved to: {output_path.absolute()}")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return output_path


def main():
    """Main execution function."""
    print("=" * 60)
    print("GDELT Embedding Enricher")
    print("=" * 60)
    print()

    data = load_data()

    enriched_data = generate_embeddings(data, model_name='all-MiniLM-L6-v2', batch_size=32)

    output_file = save_enriched_data(enriched_data)

    print()
    print("=" * 60)
    print("DONE!")
    print(f"Enriched data saved to: {output_file}")
    print(f"Total events: {len(enriched_data)}")
    print(f"Total columns: {len(enriched_data.columns)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
