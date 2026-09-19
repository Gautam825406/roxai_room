from roxroom.bots.text_normalizer import normalize_for_tts


def test_common_discourse_words_convert_to_devanagari():
    assert normalize_for_tts("kya") == "क्या"
    assert normalize_for_tts("accha") == "अच्छा"
    assert normalize_for_tts("achha") == "अच्छा"
    assert normalize_for_tts("batao") == "बताओ"
    assert normalize_for_tts("nahi") == "नहीं"


def test_punctuation_and_spacing_preserved():
    assert normalize_for_tts("kya hai?") == "क्या है?"
    assert normalize_for_tts("theek hai, chalo!") == "ठीक है, चलो!"


def test_english_loanwords_pass_through_unchanged():
    for word in ["technology", "decision", "room", "mic", "network", "data", "phone", "example"]:
        assert normalize_for_tts(word) == word


def test_ambiguous_homographs_are_deliberately_not_converted():
    # "do" (Hindi "two" / English verb), "is" (Hindi oblique "this" / English "is"),
    # "main" (Hindi "I" / English "main"), "the" (Hindi "were" / English article)
    sentence = "please do the setup, this is the main network"
    assert normalize_for_tts(sentence) == sentence


def test_acronym_and_bot_name_handling():
    assert normalize_for_tts("AI") == "ए आई"
    assert normalize_for_tts("5G network") == "फाइव जी network"
    # bot names are proper nouns, not plain Hindi words -- absent from the lexicon
    # entirely, so they never get partially transliterated
    assert normalize_for_tts("Roxstar AI Dost") == "Roxstar ए आई Dost"


def test_mixed_sentence_end_to_end():
    text = "haan bilkul, mujhe yeh samjhao ki technology kaise kaam karti hai"
    result = normalize_for_tts(text)
    assert "हाँ" in result
    assert "बिल्कुल" in result
    assert "मुझे" in result
    assert "समझाओ" in result
    assert "कैसे" in result
    assert "technology" in result  # untouched English loanword
