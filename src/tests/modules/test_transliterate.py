from src.core.config.settings import Config
from src.modules.transliterate import services
from src.modules.transliterate.services import apply_lexicon, refine, to_devanagari


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


def test_common_verb_forms_and_nouns_are_converted():
    assert apply_lexicon("Kal baarish hui to main ghar se kaam karunga, tum jaaunga mat bolna.") == (
        "कल बारिश हुई तो मैं घर से काम करूँगा, तुम जाऊँगा मत बोलना."
    )
    assert apply_lexicon("Mujhe seekhna hai, bataiye warna dekhunga") == "मुझे सीखना है, बताइए वरना देखूँगा"


def test_words_that_are_also_english_stay_in_english_text():
    for text in (
        "I ate a banana and a pair of bees flew in.",
        "The beta release is out, soya sauce and dal are on the list.",
        "Papa and mummy said the crore deal is done.",
    ):
        assert apply_lexicon(text) == text


def test_words_that_are_also_english_convert_next_to_hindi():
    assert apply_lexicon("Maine ye bana diya") == "मैंने ये बना दिया"


MIXED = "मैं कल office जाऊँगा, lekin kaam karunga."


def test_refine_is_off_by_default_and_only_for_mixed_text(monkeypatch):
    monkeypatch.setattr(services, "_ask", lambda messages, cfg: 1 / 0)
    assert refine(MIXED, Config()) == MIXED  # llm_prep off
    on = Config(llm_prep=True)
    for text in ("only english words here", "सिर्फ हिंदी शब्द यहाँ"):
        assert refine(text, on) == text  # nothing to convert


def test_refine_uses_the_models_answer_when_it_is_plausible(monkeypatch):
    monkeypatch.setattr(services, "_ask", lambda messages, cfg: "मैं कल office जाऊँगा, लेकिन काम करूँगा.")
    assert refine(MIXED, Config(llm_prep=True)) == "मैं कल office जाऊँगा, लेकिन काम करूँगा."


def test_refine_keeps_the_lexicon_text_when_the_answer_is_wrong_or_missing(monkeypatch):
    on = Config(llm_prep=True)
    for answer in (
        "",
        "मैं कल office जाऊँगा, लेकिन काम करूँगा और भी बहुत कुछ करूँगा आज.",
        "I will go to office tomorrow, but work.",
    ):
        monkeypatch.setattr(services, "_ask", lambda messages, cfg, a=answer: a)
        assert refine(MIXED, on) == MIXED


def test_prepare_chunks_without_the_llm_is_the_lexicon_with_commas_on_mid_sentence_cuts():
    chunks = [("kal ki meeting", False), ("hai.", True)]
    assert list(services.prepare_chunks(chunks, Config())) == ["कल की meeting,", "है."]


def test_prepare_chunks_leaves_the_first_chunk_to_the_lexicon_and_refines_the_rest(monkeypatch):
    asked = []
    monkeypatch.setattr(services, "refine", lambda text, cfg: asked.append(text) or text.replace("kharab", "खराब"))
    chunks = [("kal ki meeting", False), ("laptop kharab hai.", True)]
    assert list(services.prepare_chunks(chunks, Config(llm_prep=True))) == ["कल की meeting,", "laptop खराब है."]
    assert asked == ["laptop kharab है."]


def test_speakable_strips_markup_converts_roman_hindi_and_marks_mid_sentence_cuts():
    assert services.speakable("**Kal** ki meeting 😊", False, Config()) == "कल की meeting,"
    assert services.speakable("* hai.", True, Config()) == "है."


def test_speakable_is_empty_for_chunks_with_nothing_to_say():
    assert services.speakable("😊", True, Config()) == ""
    assert services.speakable("**", False, Config()) == ""


def test_speakable_keeps_markup_when_stripping_is_off():
    assert services.speakable("**hai**", True, Config(strip_markup=False)) == "**है**"
