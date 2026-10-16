from src.shared.text import split_sentences


def test_splits_on_punctuation_danda_and_newlines():
    text = "Kal milte hain. Theek hai? हाँ। Done!\n\nNext para\nline two"
    assert split_sentences(text) == ["Kal milte hain.", "Theek hai?", "हाँ।", "Done!", "Next para", "line two"]


def test_keeps_decimals_and_versions_intact():
    assert split_sentences("Version 2.5 is out. Bas.") == ["Version 2.5 is out.", "Bas."]
