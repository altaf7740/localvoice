import re
from functools import cache

import requests

from ...core.config.settings import STATIC_DIR, Config
from .prompt import EXAMPLES, SYSTEM_PROMPT

LEXICON_DIR = STATIC_DIR / "lexicon"
LATIN_WORD = re.compile(r"[A-Za-z]+")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
# a Latin word, a Devanagari word, or a single punctuation mark
TOKEN = re.compile(r"[A-Za-z]+|[ऀ-ॿ]+|[^\sA-Za-zऀ-ॿ]")


def to_devanagari(text: str, cfg: Config) -> str:
    """Rewrite Roman-script Hindi words in Devanagari so Kokoro pronounces them as Hindi."""
    if not LATIN_WORD.search(text):
        return text
    if cfg.transliterate_llm:
        text = _llm(text, cfg) or text
    return apply_lexicon(text)


def apply_lexicon(text: str) -> str:
    """Convert known Roman Hindi words. Words that are also English ("main", "me", "to") are
    converted only right after a Hindi word, or at the start of a clause before one:
    "main abhi busy hoon" -> "मैं अभी busy हूँ", but "tell me kya" keeps "me"."""
    words, ambiguous = _lexicon("hinglish.tsv"), _lexicon("hinglish_ambiguous.tsv")
    tokens = list(TOKEN.finditer(text))
    out, pos, prev = [], 0, ""  # prev = previous word as output, "" at the start of a clause
    for i, m in enumerate(tokens):
        word, low = m.group(), m.group().lower()
        new = words.get(low)
        if new is None and low in ambiguous:
            following = tokens[i + 1].group() if i + 1 < len(tokens) else ""
            if DEVANAGARI.match(prev) or (not prev and _is_hindi(following, words)):
                new = ambiguous[low]
        new = new or word
        out.append(text[pos : m.start()] + new)
        pos = m.end()
        prev = new if new[0].isalpha() else ""
    return "".join(out) + text[pos:]


def _is_hindi(word: str, words: dict[str, str]) -> bool:
    return bool(DEVANAGARI.match(word)) or word.lower() in words


@cache
def _lexicon(name: str) -> dict[str, str]:
    rows = (line.split("\t") for line in (LEXICON_DIR / name).read_text().splitlines() if line and line[0] != "#")
    return {roman: devanagari for roman, devanagari in rows}


def _llm(text: str, cfg: Config) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for roman, converted in EXAMPLES:
        messages += [{"role": "user", "content": _wrap(roman)}, {"role": "assistant", "content": converted}]
    messages.append({"role": "user", "content": _wrap(text)})
    out = _chat(messages, cfg)
    # a reply far shorter or longer than the input means the model answered or translated it
    if not out or not 0.5 <= len(out) / len(text) <= 2:
        return ""
    return out


def _wrap(text: str) -> str:
    return f"<text>{text}</text>"


def _chat(messages: list[dict], cfg: Config) -> str:
    try:
        resp = requests.post(
            f"{cfg.ollama_url}/api/chat",
            json={
                "model": cfg.transliterate_model,
                "messages": messages,
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0},
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.RequestException:
        return ""  # caller falls back to the lexicon so speech is never lost
