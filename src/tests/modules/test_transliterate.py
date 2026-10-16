from src.core.config.settings import Config
from src.modules.transliterate import services
from src.modules.transliterate.services import apply_lexicon, to_devanagari


def test_lexicon_converts_hindi_and_keeps_english():
    assert apply_lexicon("Kal ki meeting mein timeline discuss karni hai.") == (
        "कल की meeting में timeline discuss करनी है."
    )


def test_ambiguous_words_convert_only_next_to_hindi():
    assert apply_lexicon("main abhi busy hoon, baad mein call karta hoon") == ("मैं अभी busy हूँ, बाद में call करता हूँ")
    assert apply_lexicon("Please review kar do, main abhi aata hoon") == "Please review कर दो, मैं अभी आता हूँ"
    assert apply_lexicon("Can you tell me kya status hai?") == "Can you tell me क्या status है?"
    english = "The build failed because the main branch is out of date."
    assert apply_lexicon(english) == english


def test_devanagari_and_unknown_words_pass_through():
    assert apply_lexicon("मुझे लगता है server restart") == "मुझे लगता है server restart"


def test_text_without_latin_skips_the_llm(monkeypatch):
    monkeypatch.setattr(services, "_chat", lambda messages, cfg: 1 / 0)
    assert to_devanagari("कल मिलते हैं", Config()) == "कल मिलते हैं"


def test_falls_back_to_lexicon_when_ollama_is_down():
    cfg = Config(ollama_url="http://localhost:1")
    assert to_devanagari("kal milte hain", cfg) == "कल मिलते हैं"


def test_llm_output_is_used_then_lexicon_fills_gaps(monkeypatch):
    monkeypatch.setattr(services, "_chat", lambda messages, cfg: "कल dinner pe चलेंगे")
    assert to_devanagari("kal dinner pe chalenge", Config()) == "कल dinner पे चलेंगे"


def test_llm_reply_of_wrong_length_is_ignored(monkeypatch):
    monkeypatch.setattr(services, "_chat", lambda messages, cfg: "Sure! Here is the converted text: कल मिलते हैं")
    assert to_devanagari("kal milte hain", Config()) == "कल मिलते हैं"


def test_llm_disabled_never_calls_ollama(monkeypatch):
    monkeypatch.setattr(services, "_chat", lambda messages, cfg: 1 / 0)
    assert to_devanagari("kal milte hain", Config(transliterate_llm=False)) == "कल मिलते हैं"
