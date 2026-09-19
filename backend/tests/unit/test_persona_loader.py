from roxroom.bots.persona_loader import load_persona


def test_load_dost_persona():
    persona = load_persona("dost")
    assert persona.bot_id == "dost"
    assert persona.display_name == "Roxstar AI Dost"
    assert persona.gender == "male"
    assert persona.voice.provider == "sarvam"
    assert "warm, practical" in persona.system_prompt
    assert "kripya pratiksha karein" in persona.system_prompt  # banned phrase listed as banned
    assert "2 to 4 sentences" in persona.system_prompt
    assert "AI Dost, 5G kya hota hai?" in persona.system_prompt  # few-shot included


def test_load_sathi_persona():
    persona = load_persona("sathi")
    assert persona.bot_id == "sathi"
    assert persona.display_name == "Roxstar AI Sathi"
    assert persona.gender == "female"
    assert "analogies" in persona.system_prompt


def test_dost_and_sathi_prompts_are_distinguishable():
    dost = load_persona("dost")
    sathi = load_persona("sathi")
    assert dost.system_prompt != sathi.system_prompt
    assert dost.voice.voice_id != sathi.voice.voice_id


def test_both_personas_share_the_same_banned_register_rules():
    dost = load_persona("dost")
    sathi = load_persona("sathi")
    for banned in ["kripya pratiksha karein", "takneek", "nirnay lene mein saksham"]:
        assert banned in dost.system_prompt
        assert banned in sathi.system_prompt


def test_both_personas_know_how_to_resume_after_a_barge_in():
    dost = load_persona("dost")
    sathi = load_persona("sathi")
    assert "[interrupted]" in dost.system_prompt
    assert "[interrupted]" in sathi.system_prompt


def test_both_personas_have_their_own_degradation_lines():
    dost = load_persona("dost")
    sathi = load_persona("sathi")
    assert len(dost.degradation_lines) >= 1
    assert len(sathi.degradation_lines) >= 1
    assert dost.degradation_lines != sathi.degradation_lines
