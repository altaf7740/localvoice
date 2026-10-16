import re

# end of sentence: . ! ? or the Devanagari danda, followed by space; or a line break
SENTENCE_END = re.compile(r"(?<=[.!?।])\s+|\s*\n+\s*")


def split_sentences(text: str) -> list[str]:
    """Split into sentences so the first one can play while the rest are synthesised."""
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]
