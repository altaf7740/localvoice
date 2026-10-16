from src.modules.evaluation.metrics import cer, normalize


def test_normalize_keeps_devanagari_vowel_signs():
    assert normalize("मुझे लगता है, कि।") == "मुझे लगता है कि"


def test_cer_ignores_case_and_punctuation():
    assert cer("कल की Meeting है।", "कल की meeting है.") == 0.0


def test_cer_counts_character_edits():
    assert cer("abcd", "abed") == 0.25
