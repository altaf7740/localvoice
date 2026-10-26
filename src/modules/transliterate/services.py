import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import Iterator

from ...core.config.settings import STATIC_DIR, Config
from ...shared.text import as_spoken, strip_markup

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


PREP_SYSTEM = (
    "Rewrite text that mixes Hindi and English so every Hindi word is in Devanagari script and every "
    "English word stays in English letters. Keep the punctuation and the order. Do not translate, add, "
    "drop or explain anything. Output only the rewritten text."
)
PREP_EXAMPLES = (
    (
        "मैं कल office जाऊँगा, lekin baarish hui to ghar se kaam karunga.",
        "मैं कल office जाऊँगा, लेकिन बारिश हुई तो घर से काम करूँगा.",
    ),
    ("मुझे लगता है की project ka scope clear karna chahiye.", "मुझे लगता है की project का scope clear करना चाहिए."),
)
LATIN_RUN = re.compile(r"[A-Za-z]+")


def speakable(chunk: str, ends_sentence: bool, cfg: Config) -> str:
    """What the voice gets for one chunk of someone else's text (a chat model's, say): no Markdown or
    emoji, Roman Hindi in Devanagari, a comma on a mid-sentence cut. Empty if nothing is left to say."""
    text = (strip_markup(chunk) if cfg.strip_markup else chunk).strip()
    return as_spoken(to_devanagari(text), ends_sentence) if any(c.isalnum() for c in text) else ""


def prepare_chunks(chunks: list[tuple[str, bool]], cfg: Config) -> Iterator[str]:
    """What to speak for each (chunk, ends_sentence), in order. With `llm_prep` the later chunks are
    refined on a background thread while earlier ones are spoken, and the first is left as the
    lexicon made it, so speech starts at once."""
    texts = [to_devanagari(chunk) for chunk, _ in chunks]
    pool = ThreadPoolExecutor(max_workers=1) if cfg.llm_prep and len(texts) > 1 else None
    try:
        refined = [pool.submit(refine, t, cfg) for t in texts[1:]] if pool else []
        for i, ((_, ends_sentence), text) in enumerate(zip(chunks, texts)):
            yield as_spoken(refined[i - 1].result() if i and pool else text, ends_sentence)
    finally:
        if pool:
            pool.shutdown(wait=False, cancel_futures=True)  # speech was stopped: drop the waiting ones


def refine(text: str, cfg: Config) -> str:
    """`text` (already through the lexicon) with the Roman Hindi words it missed converted by an
    Ollama model. Only text that still mixes Devanagari and Latin goes to it; if Ollama is down or
    the answer drops or adds words, `text` is returned as it was."""
    if not cfg.llm_prep or not DEVANAGARI.search(text) or not LATIN_RUN.search(text):
        return text
    messages = [{"role": "system", "content": PREP_SYSTEM}]
    for before, after in PREP_EXAMPLES:
        messages += [{"role": "user", "content": before}, {"role": "assistant", "content": after}]
    out = _ask(messages + [{"role": "user", "content": text}], cfg)
    return out if _plausible(text, out) else text


def _plausible(before: str, after: str) -> bool:
    """Same number of words (give or take one), and no more Latin words than before."""
    if not after or abs(len(after.split()) - len(before.split())) > 1:
        return False
    return len(LATIN_RUN.findall(after)) <= len(LATIN_RUN.findall(before))


def _ask(messages: list[dict], cfg: Config) -> str:
    body = {
        "model": cfg.llm_prep_model,
        "messages": messages,
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0},
    }
    request = urllib.request.Request(
        f"{cfg.ollama_url}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)["message"]["content"].strip()
    except (OSError, ValueError, KeyError):
        return ""  # Ollama down or an odd reply: the caller keeps the lexicon's text
