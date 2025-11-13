#!/usr/bin/env python3
"""
GDELT Stars Visualization - Interactive 2D embedding visualization
Reduces embeddings to 2D and displays news titles as stars in space.
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_clustered_data(input_file='data/gdelt_brazil_data_clustered.csv'):
    """Load clustered GDELT data from CSV file."""
    input_path = Path(__file__).parent.parent / input_file

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Loading clustered data from: {input_path.absolute()}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} records")

    if 'x_2d' not in df.columns or 'y_2d' not in df.columns:
        raise ValueError("DataFrame must contain 'x_2d' and 'y_2d' columns. Please run enrich_embeddings.py first.")

    logger.info(f"Found 2D coordinates (x_2d, y_2d)")

    return df


def prepare_visualization_data(df):
    """
    Prepare data for visualization.

    Args:
        df: DataFrame with cluster, title, and 2D coordinate information

    Returns:
        List of dictionaries ready for JSON serialization
    """
    logger.info("Preparing visualization data...")

    vis_data = []
    for idx, row in df.iterrows():
        vis_data.append({
            'x': float(row['x_2d']),
            'y': float(row['y_2d']),
            'title': str(row.get('url_title', 'N/A')),
            'cluster': int(row.get('cluster', 0)),
            'keywords': str(row.get('cluster_keywords', '')),
            'date': str(row.get('SQLDATE', 'N/A')),
            'url': str(row.get('SOURCEURL', '')),
            'goldstein': float(row.get('GoldsteinScale', 0)) if pd.notna(row.get('GoldsteinScale')) else 0
        })

    logger.info(f"Prepared {len(vis_data)} data points")

    return vis_data


def generate_html(vis_data, output_file='output/gdelt_stars_visualization.html'):
    """Generate modern interactive HTML visualization."""
    output_path = Path(__file__).parent.parent / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cluster_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#ABEBC6'
    ]

    vis_data_json = json.dumps(vis_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GDELT Stars - Interactive Visualization</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            background: #000000;
            color: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            overflow: hidden;
            width: 100vw;
            height: 100vh;
        }}

        #canvas {{
            display: block;
            width: 100%;
            height: 100%;
            cursor: crosshair;
        }}

        #info-panel {{
            position: fixed;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            padding: 20px;
            max-width: 300px;
            backdrop-filter: blur(10px);
            z-index: 100;
        }}

        #info-panel h1 {{
            font-size: 18px;
            margin-bottom: 10px;
            color: #4ECDC4;
        }}

        #info-panel p {{
            font-size: 12px;
            line-height: 1.6;
            color: #cccccc;
            margin-bottom: 5px;
        }}

        #tooltip {{
            position: fixed;
            background: rgba(0, 0, 0, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 6px;
            padding: 12px 16px;
            pointer-events: none;
            display: none;
            z-index: 1000;
            max-width: 400px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        }}

        #tooltip.visible {{
            display: block;
        }}

        #tooltip .title {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 8px;
            color: #ffffff;
            line-height: 1.4;
        }}

        #tooltip .meta {{
            font-size: 11px;
            color: #999999;
            margin-bottom: 4px;
        }}

        #tooltip .keywords {{
            font-size: 12px;
            color: #4ECDC4;
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }}


        #controls {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            padding: 15px;
            backdrop-filter: blur(10px);
        }}

        #controls button {{
            background: rgba(78, 205, 196, 0.2);
            border: 1px solid #4ECDC4;
            color: #4ECDC4;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            margin-bottom: 8px;
            width: 100%;
            transition: all 0.2s;
        }}

        #controls button:hover {{
            background: rgba(78, 205, 196, 0.4);
        }}
    </style>
</head>
<body>
    <canvas id="canvas"></canvas>

    <div id="info-panel">
        <h1>GDELT Stars</h1>
        <p><strong>Total Events:</strong> <span id="total-events">0</span></p>
        <p><strong>Clusters:</strong> <span id="total-clusters">0</span></p>
        <p style="margin-top: 10px; font-size: 11px;">Hover over stars to see details. Scroll to zoom. Drag to pan.</p>
    </div>

    <div id="controls">
        <button onclick="resetView()">Reset View</button>
        <button onclick="toggleWords()">Toggle Words</button>
    </div>

    <div id="tooltip">
        <div class="title"></div>
        <div class="meta"></div>
        <div class="keywords"></div>
    </div>

    <script>
        const data = {vis_data_json};
        const colors = {json.dumps(cluster_colors)};

        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const tooltip = document.getElementById('tooltip');

        let width = window.innerWidth;
        let height = window.innerHeight;
        let scale = 1;
        let offsetX = 0;
        let offsetY = 0;
        let isDragging = false;
        let lastMouseX = 0;
        let lastMouseY = 0;
        let showWords = true;

        canvas.width = width * window.devicePixelRatio;
        canvas.height = height * window.devicePixelRatio;
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

        window.addEventListener('resize', () => {{
            width = window.innerWidth;
            height = window.innerHeight;
            canvas.width = width * window.devicePixelRatio;
            canvas.height = height * window.devicePixelRatio;
            canvas.style.width = width + 'px';
            canvas.style.height = height + 'px';
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
            draw();
        }});

        function worldToScreen(x, y) {{
            return {{
                x: (x * width + offsetX) * scale,
                y: (y * height + offsetY) * scale
            }};
        }}

        function screenToWorld(x, y) {{
            return {{
                x: (x / scale - offsetX) / width,
                y: (y / scale - offsetY) / height
            }};
        }}

        function calculateClusterCenters() {{
            const clusters = {{}};

            data.forEach(point => {{
                if (!clusters[point.cluster]) {{
                    clusters[point.cluster] = {{
                        x: 0,
                        y: 0,
                        count: 0,
                        keywords: point.keywords,
                        color: colors[point.cluster % colors.length]
                    }};
                }}
                clusters[point.cluster].x += point.x;
                clusters[point.cluster].y += point.y;
                clusters[point.cluster].count += 1;
            }});

            Object.keys(clusters).forEach(key => {{
                clusters[key].x /= clusters[key].count;
                clusters[key].y /= clusters[key].count;
            }});

            return clusters;
        }}

        const clusterCenters = calculateClusterCenters();

        function draw() {{
            ctx.clearRect(0, 0, width, height);

            data.forEach(point => {{
                const screen = worldToScreen(point.x, point.y);

                if (screen.x < -10 || screen.x > width + 10 ||
                    screen.y < -10 || screen.y > height + 10) {{
                    return;
                }}

                const color = colors[point.cluster % colors.length];
                const size = Math.max(2, 3 * scale);
                const glowSize = Math.max(4, 8 * scale);

                ctx.save();
                ctx.globalAlpha = 0.3;
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(screen.x, screen.y, glowSize, 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();

                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(screen.x, screen.y, size, 0, Math.PI * 2);
                ctx.fill();

                if (scale > 1.5) {{
                    const pulseSize = size + Math.sin(Date.now() / 500 + point.x * 100) * 0.5;
                    ctx.save();
                    ctx.globalAlpha = 0.5;
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.arc(screen.x, screen.y, pulseSize * 2, 0, Math.PI * 2);
                    ctx.stroke();
                    ctx.restore();
                }}
            }});

            if (showWords) {{
                Object.keys(clusterCenters).forEach(clusterId => {{
                    const cluster = clusterCenters[clusterId];
                    const screen = worldToScreen(cluster.x, cluster.y);

                    if (screen.x < -100 || screen.x > width + 100 ||
                        screen.y < -100 || screen.y > height + 100) {{
                        return;
                    }}

                    const keywords = cluster.keywords.split(',').map(k => k.trim());
                    const fontSize = Math.max(12, Math.min(24, 16 * scale));

                    ctx.save();
                    ctx.font = `${{fontSize}}px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`;
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';

                    keywords.forEach((keyword, i) => {{
                        if (keyword) {{
                            const yOffset = (i - keywords.length / 2) * (fontSize + 4);

                            ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
                            ctx.shadowBlur = 8;
                            ctx.fillStyle = cluster.color;
                            ctx.globalAlpha = 0.7;

                            ctx.fillText(keyword.toUpperCase(), screen.x, screen.y + yOffset);
                        }}
                    }});

                    ctx.restore();
                }});
            }}
        }}

        function findPointAtPosition(mouseX, mouseY) {{
            const threshold = 10;
            let closest = null;
            let closestDist = threshold;

            data.forEach(point => {{
                const screen = worldToScreen(point.x, point.y);
                const dist = Math.sqrt(
                    Math.pow(screen.x - mouseX, 2) +
                    Math.pow(screen.y - mouseY, 2)
                );

                if (dist < closestDist) {{
                    closestDist = dist;
                    closest = point;
                }}
            }});

            return closest;
        }}

        canvas.addEventListener('mousemove', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            if (isDragging) {{
                const dx = mouseX - lastMouseX;
                const dy = mouseY - lastMouseY;
                offsetX += dx / scale;
                offsetY += dy / scale;
                lastMouseX = mouseX;
                lastMouseY = mouseY;
                draw();
            }} else {{
                const point = findPointAtPosition(mouseX, mouseY);

                if (point) {{
                    tooltip.querySelector('.title').textContent = point.title;
                    tooltip.querySelector('.meta').textContent =
                        `Date: ${{point.date}} | Cluster: ${{point.cluster}} | Tone: ${{point.goldstein.toFixed(2)}}`;
                    tooltip.querySelector('.keywords').textContent =
                        `Keywords: ${{point.keywords}}`;

                    tooltip.style.left = (e.clientX + 15) + 'px';
                    tooltip.style.top = (e.clientY + 15) + 'px';
                    tooltip.classList.add('visible');

                    canvas.style.cursor = 'pointer';
                }} else {{
                    tooltip.classList.remove('visible');
                    canvas.style.cursor = isDragging ? 'grabbing' : 'crosshair';
                }}
            }}
        }});

        canvas.addEventListener('mousedown', (e) => {{
            isDragging = true;
            const rect = canvas.getBoundingClientRect();
            lastMouseX = e.clientX - rect.left;
            lastMouseY = e.clientY - rect.top;
            canvas.style.cursor = 'grabbing';
        }});

        canvas.addEventListener('mouseup', () => {{
            isDragging = false;
            canvas.style.cursor = 'crosshair';
        }});

        canvas.addEventListener('mouseleave', () => {{
            isDragging = false;
            tooltip.classList.remove('visible');
            canvas.style.cursor = 'crosshair';
        }});

        canvas.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            const world = screenToWorld(mouseX, mouseY);

            const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
            const newScale = Math.max(0.5, Math.min(10, scale * zoomFactor));

            scale = newScale;

            const newWorld = screenToWorld(mouseX, mouseY);
            offsetX += (world.x - newWorld.x) * width;
            offsetY += (world.y - newWorld.y) * height;

            draw();
        }});

        canvas.addEventListener('click', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            const point = findPointAtPosition(mouseX, mouseY);

            if (point && point.url) {{
                window.open(point.url, '_blank');
            }}
        }});

        function resetView() {{
            scale = 1;
            offsetX = 0;
            offsetY = 0;
            draw();
        }}

        function toggleWords() {{
            showWords = !showWords;
            draw();
        }}

        function initStats() {{
            const clusters = [...new Set(data.map(d => d.cluster))].sort((a, b) => a - b);
            document.getElementById('total-events').textContent = data.length;
            document.getElementById('total-clusters').textContent = clusters.length;
        }}

        function animate() {{
            draw();
            requestAnimationFrame(animate);
        }}

        initStats();
        animate();
    </script>
</body>
</html>
"""

    output_path.write_text(html_content)

    logger.info(f"Visualization saved to: {output_path.absolute()}")
    logger.info(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    return output_path


def main():
    """Main execution function."""
    print("=" * 60)
    print("GDELT Stars Visualization Generator")
    print("=" * 60)
    print()

    data = load_clustered_data()

    vis_data = prepare_visualization_data(data)

    output_file = generate_html(vis_data)

    print()
    print("=" * 60)
    print("DONE!")
    print(f"Visualization saved to: {output_file}")
    print(f"Total stars: {len(vis_data)}")
    print("=" * 60)
    print()
    print("Open the HTML file in your browser to explore the visualization!")


if __name__ == "__main__":
    main()
