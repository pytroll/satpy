"""Utilities for demo data download."""

import requests


def download_url(source, target):
    """Download a url in stream mode."""
    with requests.get(source, stream=True, timeout=10) as r:
        r.raise_for_status()
        with open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
