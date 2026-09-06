from pathlib import Path

import requests


DEFAULT_URL = (
    "https://raw.githubusercontent.com/karpathy/"
    "makemore/master/names.txt"
)


def download_words(
    url: str = DEFAULT_URL,
    cache_path: str | Path = "data/names.txt",
) -> list[str]:
    """
    Download the names dataset and return one name per line.

    The dataset is cached locally so repeated runs don't require
    another network request.
    """

    cache_path = Path(cache_path)

    if cache_path.exists():
        return cache_path.read_text().splitlines()

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    words = response.text.splitlines()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(response.text)

    return words
