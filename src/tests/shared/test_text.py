from src.shared.text import StreamChunker, as_spoken, split_chunks, split_sentences, strip_markup


def test_splits_on_punctuation_danda_and_newlines():
    text = "Kal milte hain. Theek hai? हाँ। Done!\n\nNext para\nline two"
    assert split_sentences(text) == ["Kal milte hain.", "Theek hai?", "हाँ।", "Done!", "Next para", "line two"]


def test_keeps_decimals_and_versions_intact():
    assert split_sentences("Version 2.5 is out. Bas.") == ["Version 2.5 is out.", "Bas."]


def chunks(text):
    return [c for c, _ in split_chunks(text)]


def test_cuts_at_commas_and_sentence_ends():
    assert split_chunks("Python ek popular language hai, jo sikhne mein easy hai. Bas itna.") == [
        ("Python ek popular language hai,", False),
        ("jo sikhne mein easy hai.", True),
        ("Bas itna.", True),
    ]


def test_cuts_before_clause_connectors_but_not_joining_words():
    text = "Meeting kal subah ko office mein hogi lekin location abhi tak confirm nahi hui hai"
    assert chunks(text) == ["Meeting kal subah ko office mein hogi", "lekin location abhi tak confirm nahi hui hai"]
    joined = "Python ke baare mein bahut kuch seekhna hai aur uska use bhi karna hai"
    assert chunks(joined) == [joined]


def test_short_first_piece_is_not_cut_off_alone():
    assert chunks("Haan, theek hai, chalo kal milte hain.") == ["Haan, theek hai, chalo kal milte hain."]
    assert chunks("हाँ। ठीक है।") == ["हाँ। ठीक है।"]


def test_keeps_abbreviations_and_numbers_whole():
    assert chunks("Dr. Sharma ne version 3.5 use kiya.") == ["Dr. Sharma ne version 3.5 use kiya."]


def test_each_line_is_its_own_chunk():
    assert chunks("Heading\n\nfirst line\nsecond line") == ["Heading", "first line", "second line"]


def test_long_run_without_marks_is_cut_at_a_space():
    text = " ".join(["shabd"] * 40)
    parts = chunks(text)
    assert len(parts) > 1 and all(len(p) <= 150 for p in parts) and " ".join(parts) == text


def test_mid_sentence_cut_is_spoken_with_a_comma():
    assert as_spoken("kal subah hogi", False) == "kal subah hogi,"
    assert as_spoken("kal subah hogi,", False) == "kal subah hogi,"
    assert as_spoken("kal subah hogi.", True) == "kal subah hogi."


def test_strip_markup_removes_marks_and_emoji_but_not_plain_text():
    assert strip_markup("**Bold** and *soft* and `code` 😊") == "Bold and soft and code "
    assert strip_markup("## Heading\n* one\n- two\n1. three\n> quote") == "Heading\none\ntwo\nthree\nquote"
    assert strip_markup("See [the docs](http://x.io/a_b) now") == "See the docs now"
    assert strip_markup("2*3*4 = 24, snake_case, a - b") == "2*3*4 = 24, snake_case, a - b"


def stream(text, step):
    chunker, out = StreamChunker(), []
    for i in range(0, len(text), step):
        out += chunker.feed(text[i : i + step])
    return out + chunker.flush()


def test_streaming_cuts_the_same_way_whatever_size_the_pieces_are():
    text = (
        "Python ek popular language hai, jo sikhne mein easy hai. Dr. Sharma ne 3.5 use kiya.\n"
        "Meeting kal subah ko office mein hogi lekin location abhi tak confirm nahi hui hai"
    )
    assert stream(text, 1) == stream(text, 7) == stream(text, 1000) == split_chunks(text) != []


def test_first_cut_starts_speech_at_a_space_only_for_the_first_chunk():
    text = "Meeting mein project delay ke baare mein bolne ke liye ye points yaad rakhna aur phir baaki sab batana"
    plain = stream(text, 3)
    early = []
    chunker = StreamChunker(first_cut=24)
    for i in range(0, len(text), 3):
        early += chunker.feed(text[i : i + 3])
    early += chunker.flush()
    assert len(plain) == 1  # nothing to cut at without punctuation
    first, ends_sentence = early[0]
    assert not ends_sentence and 12 < len(first) <= 24 and text.startswith(first + " ")  # whole words only
    assert " ".join(c for c, _ in early) == text


def test_first_cut_does_not_leave_a_dangling_word():
    text = "Hello there, I am Gemma an AI assistant here to help you with many things today."
    chunker = StreamChunker(first_cut=22)
    first = [c for t in range(0, len(text), 2) for c in chunker.feed(text[t : t + 2])][0][0]
    assert text.startswith(first)
    assert first == "Hello there,"  # not "Hello there, I am"
