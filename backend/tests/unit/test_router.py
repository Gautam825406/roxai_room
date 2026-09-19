from roxroom.orchestrator.bot_aliases import find_bot_mentions
from roxroom.orchestrator.router import Router, classify_persona_affinity


def test_find_bot_mentions_exact_names():
    mentions = find_bot_mentions("Dost aur Sathi dono sun rahe ho?")
    assert [m.bot for m in mentions] == ["dost", "sathi"]


def test_find_bot_mentions_tolerates_known_asr_variants():
    assert [m.bot for m in find_bot_mentions("dosth kaisa hai")] == ["dost"]
    assert [m.bot for m in find_bot_mentions("dast tum batao")] == ["dost"]
    assert [m.bot for m in find_bot_mentions("saathi ek example do")] == ["sathi"]
    assert [m.bot for m in find_bot_mentions("sathee suno")] == ["sathi"]


def test_find_bot_mentions_does_not_false_positive_on_common_words():
    for text in ["yeh sabse best hai", "iska cost kya hai", "wo sabse most popular hai"]:
        assert find_bot_mentions(text) == []


def test_persona_affinity_definitional_goes_to_dost():
    assert classify_persona_affinity("yeh kya hota hai, define karo") == "dost"


def test_persona_affinity_example_goes_to_sathi():
    assert classify_persona_affinity("mujhe samajh nahi aa raha, ek example do") == "sathi"


def test_persona_affinity_neutral_text_returns_none():
    assert classify_persona_affinity("mujhe cricket pasand hai") is None


def test_route_single_explicit_address():
    router = Router()
    decision = router.route(text="AI Dost, tumhara naam kya hai?", stage_b_addressed_bot="none")
    assert decision.rule_fired == "explicit_address"
    assert decision.chosen_bot == "dost"


def test_route_multi_bot_plan_ordered_by_mention_position_with_fragments():
    router = Router()
    text = "AI Dost tum pehle answer karo, phir AI Sathi ek example dena"
    decision = router.route(text=text, stage_b_addressed_bot="none")
    assert decision.rule_fired == "multi_bot_plan"
    assert [step.bot for step in decision.plan] == ["dost", "sathi"]
    assert "answer karo" in decision.plan[0].instruction
    assert "example dena" in decision.plan[1].instruction


def test_route_stage_b_addressed_bot_used_when_no_explicit_mention():
    router = Router()
    decision = router.route(text="kal match kaisa raha?", stage_b_addressed_bot="sathi")
    assert decision.rule_fired == "stage_b_addressed"
    assert decision.chosen_bot == "sathi"


def test_route_persona_affinity_fallback():
    router = Router()
    decision = router.route(text="ek simple example do na", stage_b_addressed_bot="none")
    assert decision.rule_fired == "persona_affinity"
    assert decision.chosen_bot == "sathi"


def test_route_alternation_fallback_picks_least_recently_spoken():
    router = Router()
    router.record_reply_completed("dost", at=10.0)
    router.record_reply_completed("sathi", at=5.0)
    decision = router.route(text="kuch bhi neutral statement", stage_b_addressed_bot="none")
    assert decision.rule_fired == "alternation_fallback"
    assert decision.chosen_bot == "sathi"  # sathi spoke longer ago (5.0 < 10.0)


def test_route_alternation_fallback_defaults_when_neither_has_spoken():
    router = Router()
    decision = router.route(text="kuch bhi neutral statement", stage_b_addressed_bot="none")
    assert decision.rule_fired == "alternation_fallback"
    assert decision.chosen_bot in ("dost", "sathi")
