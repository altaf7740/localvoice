import re
from functools import cache

from ...core.config.settings import STATIC_DIR

LEXICON_DIR = STATIC_DIR / "lexicon"
LATIN_WORD = re.compile(r"[A-Za-z]+")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
# a Latin word, a Devanagari word, or a single punctuation mark
TOKEN = re.compile(r"[A-Za-z]+|[ऀ-ॿ]+|[^\sA-Za-zऀ-ॿ]")


def to_devanagari(text: str) -> str:
    """Rewrite Roman-script Hindi words in Devanagari so the TTS engine pronounces them as Hindi."""
    if not LATIN_WORD.search(text):
        return text
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
