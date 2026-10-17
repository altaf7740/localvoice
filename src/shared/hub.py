from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


def cached(repo: str, allow_patterns: list[str] | None = None) -> Path | str:
    """The downloaded copy of `repo`, or `repo` itself while it isn't fully downloaded (so the loader
    fetches it, with progress bars). Loading the local copy skips asking the Hub for updates, and its
    "Fetching N files" bar, on every start, and works offline."""
    try:
        return Path(snapshot_download(repo, local_files_only=True, allow_patterns=allow_patterns))
    except LocalEntryNotFoundError:  # also raised when a download was interrupted
        return repo
