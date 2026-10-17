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


def test_to_devanagari_passes_devanagari_through_and_converts_roman_hindi():
    assert to_devanagari("कल मिलते हैं") == "कल मिलते हैं"
    assert to_devanagari("kal milte hain") == "कल मिलते हैं"
