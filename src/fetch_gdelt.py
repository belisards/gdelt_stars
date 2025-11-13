#!/usr/bin/env python3
"""
GDELT Data Fetcher - Downloads Brazil data and enriches URLs with titles
Filters for democracy-related CAMEO event codes
Saves data to CSV file for visualization.
"""

import pandas as pd
import gdelt
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from typing import Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import StringIO

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

_shared_session = None

DEMOCRACY_CATEGORIES = {
    'Political Repression & Restrictions': [
        '172', '1721', '1722', '1723', '1724', '173', '174', '175'
    ],
    'Protest & Dissent Events': [
        '140', '141', '1411', '1412', '1413', '1414', '143', '145',
        '1451', '1452', '1453', '1454'
    ],
    'Threats to Democratic Order': [
        '132', '1321', '1322', '1324', '137'
    ],
    'Demands for Democratic Reform': [
        '104', '1041', '1042', '1043', '1044'
    ],
    'Rejection of Democratic Processes': [
        '123', '1231', '1232', '1233', '1234', '128'
    ],
    'Violence Against Civilians': [
        '180', '181', '182', '1822', '1823', '185', '186'
    ],
    'Mass Violence & Persecution': [
        '201', '202', '203'
    ],
    'Judicial & Legal Actions': [
        '092', '112', '1122', '115', '116'
    ],
    'Electoral & Political Cooperation/Conflict': [
        '0241', '0244', '0831', '0832', '0833', '0834', '161'
    ],
    'Media & Information Control Related': [
        '011', '111', '113'
    ]
}

DEMOCRACY_EVENT_CODES = set()
CODE_TO_CATEGORY = {}
for category, codes in DEMOCRACY_CATEGORIES.items():
    for code in codes:
        DEMOCRACY_EVENT_CODES.add(code)
        CODE_TO_CATEGORY[code] = category


def _get_shared_session():
    """Get or create a shared session with optimized settings."""
    global _shared_session
    if _shared_session is None:
        _shared_session = requests.Session()
        retry = Retry(
            total=0,
            backoff_factor=0,
            status_forcelist=[]
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=20,
            pool_maxsize=50
        )
        _shared_session.mount('http://', adapter)
        _shared_session.mount('https://', adapter)
        _shared_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    return _shared_session


def get_page_title(url: str, timeout: int = 5, session: Optional[requests.Session] = None) -> Optional[str]:
    """
    Fetch the title of a web page from a given URL.

    Args:
        url: The URL to fetch the title from
        timeout: Request timeout in seconds (default: 5)
        session: Optional requests.Session to use (default: uses shared session)

    Returns:
        The page title if found, None otherwise
    """
    if not url or not isinstance(url, str):
        return None

    if session is None:
        session = _get_shared_session()

    try:
        response = session.get(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()

        try:
            soup = BeautifulSoup(response.content, 'lxml')
        except:
            soup = BeautifulSoup(response.content, 'html.parser')

        title_tag = soup.find('title')

        if title_tag and title_tag.string:
            title = title_tag.string.strip()
            return title if title else None

        return None

    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching title from {url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error fetching title from {url}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error fetching title from {url}: {e}")
        return None


def filter_democracy_events(df):
    """
    Filter GDELT data to include only democracy-related CAMEO event codes.

    Args:
        df: DataFrame with EventCode column

    Returns:
        DataFrame filtered to democracy-related events with added category column
    """
    if df.empty:
        return df

    df = df.copy()

    df['EventCode'] = df['EventCode'].astype(str)

    mask = df['EventCode'].isin(DEMOCRACY_EVENT_CODES)
    df_filtered = df[mask]

    df_filtered['democracy_category'] = df_filtered['EventCode'].map(CODE_TO_CATEGORY)

    logger.info(f"Filtered to {len(df_filtered)} democracy-related events from {len(df)} total events")

    return df_filtered


def fetch_brazil_data(days=7):
    """Fetch GDELT data for Brazil for the last N days and filter for democracy events."""
    logger.info(f"Fetching GDELT data for Brazil (last {days} days)...")

    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')

    logger.info(f"Date range: {start_date} to {end_date}")

    gd = gdelt.gdelt(version=2)
    df = gd.Search([start_date, end_date], table='events', coverage=True)

    if df is None or df.empty:
        logger.warning("No data retrieved")
        return pd.DataFrame()

    logger.info(f"Total records downloaded: {len(df)}")

    mask = df['Actor1Code'].apply(
        lambda x: str(x).startswith('BR') if pd.notna(x) else False
    )
    df_brazil = df[mask]

    logger.info(f"Brazil records: {len(df_brazil)}")

    df_brazil_democracy = filter_democracy_events(df_brazil)

    return df_brazil_democracy


def enrich_urls_with_titles(df, max_workers=20):
    """
    Enrich DataFrame with URL titles.

    Args:
        df: DataFrame with SOURCEURL column
        max_workers: Number of concurrent workers for fetching titles

    Returns:
        DataFrame with added 'url_title' column
    """
    if df.empty:
        return df

    df = df.copy()

    unique_urls = df['SOURCEURL'].dropna().unique()
    total_urls = len(unique_urls)

    logger.info(f"Fetching titles for {total_urls} unique URLs using {max_workers} workers...")

    url_titles = {}
    successful_fetches = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(get_page_title, url): url for url in unique_urls}

        completed = 0
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            completed += 1

            if completed % 100 == 0:
                logger.info(f"Progress: {completed}/{total_urls} URLs processed")

            try:
                title = future.result()
                url_titles[url] = title
                if title:
                    successful_fetches += 1
            except Exception as e:
                logger.warning(f"Error fetching title for {url}: {e}")
                url_titles[url] = None

    df['url_title'] = df['SOURCEURL'].map(url_titles)

    logger.info(f"Successfully fetched {successful_fetches}/{total_urls} titles ({successful_fetches/total_urls*100:.1f}%)")

    return df


def save_data(df, output_file='data/gdelt_brazil_data.csv'):
    """Save DataFrame to CSV file."""
    output_path = Path(__file__).parent.parent / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    logger.info(f"Data saved to: {output_path.absolute()}")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return output_path


def main():
    """Main execution function."""
    print("=" * 60)
    print("GDELT Brazil Data Fetcher (Democracy Events)")
    print("=" * 60)
    print()

    data = fetch_brazil_data(days=7)

    if data.empty:
        logger.error("No data to process")
        return

    enriched_data = enrich_urls_with_titles(data, max_workers=20)

    output_file = save_data(enriched_data)

    print()
    print("=" * 60)
    print("DONE!")
    print(f"Data saved to: {output_file}")
    print(f"Total events: {len(enriched_data)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
