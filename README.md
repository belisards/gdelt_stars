# GDELT Stars

Interactive visualization of GDELT news data as stars in space, grouped by semantic similarity.

## Project Structure

```
gdelt_stars/
├── src/                        # Python source code
│   ├── fetch_gdelt.py         # Fetch and enrich GDELT data
│   ├── enrich_embeddings.py   # Generate embeddings and 2D coordinates
│   ├── cluster_analysis.py    # Cluster data and extract keywords
│   ├── visualize_stars.py     # Generate interactive visualization
│   └── run_pipeline.py        # Master pipeline script
├── data/                       # Data files (gitignored)
│   ├── gdelt_brazil_data.csv
│   ├── gdelt_brazil_data_enriched.csv
│   └── gdelt_brazil_data_clustered.csv
├── output/                     # Visualization outputs
│   └── gdelt_stars_visualization.html
└── pyproject.toml             # Project dependencies
```

## Installation

```bash
uv sync
```

## Usage

### Run Complete Pipeline

```bash
uv run python src/run_pipeline.py
```

### Run Individual Steps

1. **Fetch GDELT data:**
   ```bash
   uv run python src/fetch_gdelt.py
   ```

2. **Generate embeddings:**
   ```bash
   uv run python src/enrich_embeddings.py
   ```

3. **Perform clustering:**
   ```bash
   uv run python src/cluster_analysis.py
   ```

4. **Generate visualization:**
   ```bash
   uv run python src/visualize_stars.py
   ```

## Visualization Features

- **Interactive canvas** with pan (drag) and zoom (scroll)
- **Stars** representing news events, positioned by semantic similarity
- **Color-coded clusters** with spatial coherence
- **Floating keywords** at cluster centers
- **Hover tooltips** showing title, date, cluster, and sentiment
- **Click stars** to open source URLs
- **Toggle keywords** on/off

## Dependencies

- pandas
- gdelt
- beautifulsoup4
- lxml
- requests
- sentence-transformers
- scikit-learn
