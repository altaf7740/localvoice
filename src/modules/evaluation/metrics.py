import re


def normalize(text: str) -> str:
    """Lowercase, drop punctuation (incl. the danda) and collapse whitespace."""
    # not [^\w\s]: Devanagari vowel signs (ि, ा) are not \w and would be stripped
    text = re.sub(r"[.,!?;:'\"()\[\]…।॥-]", " ", text.lower())
    return " ".join(text.split())


def cer(ref: str, hyp: str) -> float:
    """Character error rate between a mixed-script reference and a Whisper transcript."""
    r, h = normalize(ref), normalize(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)
