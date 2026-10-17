from pathlib import Path

from huggingface_hub.errors import LocalEntryNotFoundError

from src.shared import hub


def test_cached_returns_the_local_copy(monkeypatch):
    calls = []

    def fake_download(repo, **kwargs):
        calls.append(kwargs)
        return "/hub/models--a--b/snapshots/123"

    monkeypatch.setattr(hub, "snapshot_download", fake_download)
    assert hub.cached("a/b", ["*.json"]) == Path("/hub/models--a--b/snapshots/123")
    assert calls == [{"local_files_only": True, "allow_patterns": ["*.json"]}]


def test_cached_falls_back_to_the_repo_when_not_downloaded(monkeypatch):
    def fake_download(repo, **kwargs):
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(hub, "snapshot_download", fake_download)
    assert hub.cached("a/b") == "a/b"
