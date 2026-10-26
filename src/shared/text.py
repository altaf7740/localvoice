import re

# end of sentence: . ! ? or the Devanagari danda, followed by space; or a line break
SENTENCE_END = re.compile(r"(?<=[.!?।])\s+|\s*\n+\s*")


def split_sentences(text: str) -> list[str]:
    """Split into sentences so the first one can play while the rest are synthesised."""
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


# end of a sentence inside a line: . ! ? । … (and closing quotes) followed by a space
CLAUSE_END = re.compile(r"[.!?।॥…]+[\"')\]”’]*(?=\s)")
PHRASE_MARKS = ",;:،—–"
PHRASE_END = re.compile(rf"[{PHRASE_MARKS}](?=\s)")
# clause-level connectors only: joining words (और, या, तो, कि, and, or, so, that) sit inside a phrase
CONNECTORS = (
    "लेकिन",
    "परंतु",
    "मगर",
    "क्योंकि",
    "इसलिए",
    "जबकि",
    "lekin",
    "magar",
    "kyunki",
    "kyuki",
    "isliye",
    "jabki",
    "but",
    "because",
    "while",
    "however",
)
CONNECTOR = re.compile(r"(?<=\s)(?:%s)(?=\s)" % "|".join(CONNECTORS), re.IGNORECASE)
ABBREVIATIONS = ("mr", "mrs", "ms", "dr", "prof", "vs", "etc", "e.g", "i.e")
MIN_SENTENCE = 8  # a shorter one ("हाँ।") joins the next
# smallest piece cut at a comma or a connector; the first is short so speech starts early
FIRST_PHRASE, PHRASE = 22, 38
FIRST_CONNECTOR, CONNECTOR_MIN = 30, 45
MAX_CHUNK = 150  # a run with no mark at all is cut at the last space before this
# words a phrase can't end on ("Gemma, an | AI assistant"): an early cut backs up past them
DANGLING = frozenset(
    "a an the of to in on at for and or but with from by as if so that than my your our his her their its "
    "i am is are was were be been "
    "ka ki ke ko se mein me par pe aur ya ek ye yeh wo woh toh to "
    "का की के को से में पर और या एक यह वह तो कि इस उस इन उन ये वो मैं हम तुम आप".split()
)


def split_chunks(text: str) -> list[tuple[str, bool]]:
    """Split into phrases, as (text, ends_a_sentence): at sentence ends, at commas and similar marks,
    and before clause-level connectors ("lekin", "but"), never inside a number or abbreviation.
    The first can play while the rest are synthesised."""
    chunks: list[tuple[str, bool]] = []
    for line in text.splitlines():
        rest = line.strip()
        while rest:
            end, ends_sentence = _find_cut(rest, first=not chunks) or (len(rest), True)
            chunk, rest = rest[:end].strip(), rest[end:].strip()
            if chunk:
                chunks.append((chunk, ends_sentence))
    return chunks


def as_spoken(chunk: str, ends_sentence: bool) -> str:
    """A comma on a mid-sentence cut keeps the voice's tone open instead of ending a sentence."""
    return chunk if ends_sentence or chunk[-1:] in PHRASE_MARKS else chunk + ","


def _find_cut(text: str, first: bool) -> tuple[int, bool] | None:
    """(length of the first chunk, whether it ends a sentence), or None if `text` has no cut yet."""
    best = None
    for m in CLAUSE_END.finditer(text):
        word = re.findall(r"[\w.]+$", text[: m.start()])
        if word and word[0].lower().rstrip(".") in ABBREVIATIONS:
            continue
        if m.end() >= MIN_SENTENCE:
            best = (m.end(), True)
            break
    for pattern, at_end, floor in (
        (PHRASE_END, True, FIRST_PHRASE if first else PHRASE),
        (CONNECTOR, False, FIRST_CONNECTOR if first else CONNECTOR_MIN),
    ):
        for m in pattern.finditer(text):
            pos = m.end() if at_end else m.start()
            if pos >= floor and (best is None or pos < best[0]):
                best = (pos, False)
                break
    if best is None and len(text) > MAX_CHUNK and " " in text[:MAX_CHUNK]:
        best = (text.rfind(" ", 0, MAX_CHUNK), False)
    return best


_EMOJI = re.compile("[\U0001f000-\U0001faff\u2600-\u27bf\ufe0f\u200d]")
_MARKUP = (
    (re.compile(r"!?\[([^\]]*)\]\([^)]*\)"), r"\1"),  # [text](url) -> text
    (re.compile(r"(\*\*|__)(.+?)\1"), r"\2"),  # **bold**, __bold__
    (re.compile(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])"), r"\1"),  # *italic*, not 2*3*4
    (re.compile(r"`+([^`]*)`+"), r"\1"),  # `code`
    (re.compile(r"^\s*(?:#{1,6}|>|[-*+•]|\d{1,2}[.)])\s+", re.MULTILINE), ""),  # heading, quote, bullet, "1."
)


def strip_markup(text: str) -> str:
    """Drop Markdown marks and emoji that a voice would read out or stumble on ("**", "#", "😊")."""
    for pattern, replacement in _MARKUP:
        text = pattern.sub(replacement, text)
    return _EMOJI.sub("", text)


class StreamChunker:
    """split_chunks() for text that arrives in pieces (an LLM's tokens): feed() returns the
    (chunk, ends_sentence) pairs completed so far, and flush() the rest once the text has ended.
    `first_cut`: if the first chunk reaches this many characters with no cut point, cut it at a
    space so speech starts sooner (0 = wait for punctuation)."""

    def __init__(self, first_cut: int = 0):
        self.buf = ""
        self.count = 0
        self.first_cut = first_cut

    def feed(self, text: str) -> list[tuple[str, bool]]:
        self.buf += text
        out = []
        while True:
            line, newline, after = self.buf.partition("\n")
            cut = _find_cut(line, first=self.count == 0)
            if cut is None and self.count == 0 and self.first_cut and len(line.strip()) >= self.first_cut:
                space = line.rstrip().rfind(" ")
                while space > self.first_cut // 2 and line[:space].split()[-1].lower().strip(",;:") in DANGLING:
                    space = line.rfind(" ", 0, space)
                cut = (space, False) if space > self.first_cut // 2 else None
            if cut is None and not newline:
                return out  # no cut yet, and the line isn't over
            end, ends_sentence = cut or (len(line), True)
            chunk = line[:end].strip()
            self.buf = line[end:] + newline + after if cut else after
            if chunk:
                out.append((chunk, ends_sentence))
                self.count += 1

    def flush(self) -> list[tuple[str, bool]]:
        rest, self.buf = self.buf.strip(), ""
        return [(rest, True)] if rest else []
