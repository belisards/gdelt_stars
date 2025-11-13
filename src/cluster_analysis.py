#!/usr/bin/env python3
"""
GDELT Cluster Analysis - Assigns clusters and extracts keywords
Reads enriched CSV file and performs clustering on embeddings.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_enriched_data(input_file='data/gdelt_brazil_data_enriched.csv'):
    """Load enriched GDELT data from CSV file."""
    input_path = Path(__file__).parent.parent / input_file

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Loading enriched data from: {input_path.absolute()}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")

    if 'x_2d' not in df.columns or 'y_2d' not in df.columns:
        raise ValueError("DataFrame must contain 'x_2d' and 'y_2d' columns")

    return df


def extract_2d_coordinates(df):
    """Extract 2D coordinates from DataFrame."""
    logger.info("Extracting 2D coordinates for clustering...")

    coords_2d = df[['x_2d', 'y_2d']].values
    logger.info(f"2D coordinates shape: {coords_2d.shape}")

    return coords_2d


def perform_clustering(coords_2d, n_clusters=10, random_state=42):
    """
    Perform K-means clustering on 2D coordinates.

    Args:
        coords_2d: Numpy array of 2D coordinates
        n_clusters: Number of clusters to create
        random_state: Random seed for reproducibility

    Returns:
        Array of cluster labels
    """
    logger.info(f"Performing K-means clustering on 2D coordinates with {n_clusters} clusters...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(coords_2d)

    logger.info(f"Clustering complete. Cluster distribution:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        logger.info(f"  Cluster {cluster_id}: {count} records")

    return cluster_labels


def extract_keywords_from_text(text, n_keywords=3):
    """
    Extract keywords from text using simple word frequency.

    Args:
        text: Text to extract keywords from
        n_keywords: Number of top keywords to return

    Returns:
        List of top keywords
    """
    if not isinstance(text, str) or not text.strip():
        return []

    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)

    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
        'those', 'it', 'its', 'they', 'them', 'their', 'what', 'which',
        'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
        'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just'
    }

    words = [w for w in text.split() if len(w) > 2 and w not in stopwords]

    word_counts = Counter(words)
    top_keywords = [word for word, count in word_counts.most_common(n_keywords)]

    return top_keywords


def extract_cluster_keywords(df, cluster_col='cluster', n_keywords=3):
    """
    Extract top keywords for each cluster.

    Args:
        df: DataFrame with cluster assignments and url_title
        cluster_col: Name of cluster column
        n_keywords: Number of keywords per cluster

    Returns:
        DataFrame with cluster keywords
    """
    logger.info(f"Extracting top {n_keywords} keywords for each cluster...")

    cluster_keywords = {}

    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster_id]

        all_text = ' '.join(
            cluster_data['url_title']
            .fillna('')
            .astype(str)
            .tolist()
        )

        keywords = extract_keywords_from_text(all_text, n_keywords=n_keywords)

        cluster_keywords[cluster_id] = ', '.join(keywords) if keywords else 'N/A'

        logger.info(f"  Cluster {cluster_id}: {cluster_keywords[cluster_id]}")

    df['cluster_keywords'] = df[cluster_col].map(cluster_keywords)

    return df


def save_clustered_data(df, output_file='data/gdelt_brazil_data_clustered.csv'):
    """Save clustered DataFrame to CSV file."""
    output_path = Path(__file__).parent.parent / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    logger.info(f"Clustered data saved to: {output_path.absolute()}")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return output_path


def main():
    """Main execution function."""
    print("=" * 60)
    print("GDELT Cluster Analysis (2D)")
    print("=" * 60)
    print()

    data = load_enriched_data()

    coords_2d = extract_2d_coordinates(data)

    cluster_labels = perform_clustering(coords_2d, n_clusters=10, random_state=42)

    data['cluster'] = cluster_labels

    clustered_data = extract_cluster_keywords(data, cluster_col='cluster', n_keywords=3)

    output_file = save_clustered_data(clustered_data)

    print()
    print("=" * 60)
    print("DONE!")
    print(f"Clustered data saved to: {output_file}")
    print(f"Total events: {len(clustered_data)}")
    print(f"Total clusters: {clustered_data['cluster'].nunique()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
