import pytest

from roxroom.context.extractor import ExtractionResult
from roxroom.context.fakes import FailingExtractor, FakeExtractor, FakeSummarizer
from roxroom.context.models import Turn
from roxroom.context.room_context import (
    COMPRESS_EVERY_N_TURNS,
    MAX_BUFFERED_TURNS,
    PROMPT_RECENT_TURNS,
    RoomContext,
)


@pytest.mark.asyncio
async def test_prompt_context_returns_all_turns_when_under_the_cap():
    ctx = RoomContext()
    for i in range(9):  # stays under COMPRESS_EVERY_N_TURNS, so nothing gets compressed away
        await ctx.add_turn(
            speaker_id="p1", display_name="Rahul", role="human", text=f"turn {i}", ts=float(i)
        )
    pc = ctx.build_prompt_context()
    assert [t.text for t in pc.recent_turns] == [f"turn {i}" for i in range(9)]


def test_prompt_context_caps_at_last_12_turns():
    ctx = RoomContext()
    for i in range(15):
        ctx.turns.append(
            Turn(speaker_id="p1", display_name="Rahul", role="human", text=f"turn {i}", ts=float(i))
        )
    pc = ctx.build_prompt_context()
    assert len(pc.recent_turns) == PROMPT_RECENT_TURNS
    assert [t.text for t in pc.recent_turns] == [f"turn {i}" for i in range(3, 15)]


@pytest.mark.asyncio
async def test_speaker_profiles_do_not_leak_between_speakers():
    """Scenario-5-style check: Rahul's name/cricket fact must survive and must never
    attach to Priya's profile even though they're speaking in the same room."""
    extractor = FakeExtractor(
        by_text={
            "mera naam Rahul hai aur mujhe cricket pasand hai": ExtractionResult(
                facts=["name is Rahul"], preferences=["likes cricket"], entities=["cricket"]
            ),
            "mujhe toh painting pasand hai": ExtractionResult(preferences=["likes painting"]),
        }
    )
    ctx = RoomContext(extractor=extractor)

    await ctx.add_turn(
        speaker_id="rahul",
        display_name="Rahul",
        role="human",
        text="mera naam Rahul hai aur mujhe cricket pasand hai",
        ts=0.0,
    )
    await ctx.add_turn(
        speaker_id="priya",
        display_name="Priya",
        role="human",
        text="mujhe toh painting pasand hai",
        ts=1.0,
    )

    rahul = ctx.get_speaker_profile("rahul")
    priya = ctx.get_speaker_profile("priya")

    assert rahul.stated_facts == ["name is Rahul"]
    assert rahul.preferences == ["likes cricket"]
    assert priya.stated_facts == []
    assert priya.preferences == ["likes painting"]
    assert "likes cricket" not in priya.preferences


@pytest.mark.asyncio
async def test_bot_turns_do_not_trigger_extraction():
    extractor = FakeExtractor()
    ctx = RoomContext(extractor=extractor)
    await ctx.add_turn(speaker_id="dost", display_name="Roxstar AI Dost", role="bot", text="haan bilkul", ts=0.0)
    assert extractor.calls == []
    assert ctx.get_speaker_profile("dost") is None


@pytest.mark.asyncio
async def test_recent_entities_are_capped_at_3_and_move_to_end_on_repeat():
    extractor = FakeExtractor(
        by_text={
            "t1": ExtractionResult(entities=["cricket"]),
            "t2": ExtractionResult(entities=["Rahul"]),
            "t3": ExtractionResult(entities=["IPL"]),
            "t4": ExtractionResult(entities=["Mumbai"]),
            "t5": ExtractionResult(entities=["cricket"]),  # re-mention -> moves to end
        }
    )
    ctx = RoomContext(extractor=extractor)
    for i, text in enumerate(["t1", "t2", "t3", "t4", "t5"]):
        await ctx.add_turn(speaker_id="p1", display_name="Rahul", role="human", text=text, ts=float(i))

    assert list(ctx.recent_entities) == ["IPL", "Mumbai", "cricket"]
    pc = ctx.build_prompt_context()
    assert pc.recent_entities_line() == "RECENT ENTITIES: IPL, Mumbai, cricket"


def test_recent_entities_line_empty_when_no_entities():
    ctx = RoomContext()
    assert ctx.build_prompt_context().recent_entities_line() == ""


@pytest.mark.asyncio
async def test_compression_fires_every_10_turns_and_shrinks_buffer():
    summarizer = FakeSummarizer()
    ctx = RoomContext(summarizer=summarizer)
    for i in range(COMPRESS_EVERY_N_TURNS):
        await ctx.add_turn(
            speaker_id="p1", display_name="Rahul", role="human", text=f"turn {i}", ts=float(i)
        )

    assert len(summarizer.calls) == 1
    compressed_turns, previous_summary = summarizer.calls[0]
    assert previous_summary == ""
    assert [t.text for t in compressed_turns] == ["turn 0", "turn 1", "turn 2", "turn 3", "turn 4"]
    # 10 added, oldest 5 compressed away -> 5 remain buffered
    assert len(ctx.turns) == 5
    assert ctx.rolling_summary != ""


@pytest.mark.asyncio
async def test_buffer_never_exceeds_hard_cap_under_sustained_load():
    ctx = RoomContext(summarizer=FakeSummarizer())
    for i in range(50):
        await ctx.add_turn(
            speaker_id="p1", display_name="Rahul", role="human", text=f"turn {i}", ts=float(i)
        )
        assert len(ctx.turns) <= MAX_BUFFERED_TURNS


@pytest.mark.asyncio
async def test_text_chat_and_voice_share_the_same_turn_buffer():
    ctx = RoomContext()
    await ctx.add_turn(
        speaker_id="p1", display_name="Rahul", role="human", text="voice hello", ts=0.0, modality="voice"
    )
    await ctx.add_turn(
        speaker_id="p1", display_name="Rahul", role="human", text="text hello", ts=1.0, modality="text"
    )
    pc = ctx.build_prompt_context()
    assert [t.modality for t in pc.recent_turns] == ["voice", "text"]


@pytest.mark.asyncio
async def test_add_turn_survives_a_failing_extractor():
    extractor = FailingExtractor()
    ctx = RoomContext(extractor=extractor)

    turn = await ctx.add_turn(
        speaker_id="rahul", display_name="Rahul", role="human", text="mera naam Rahul hai", ts=0.0
    )

    assert turn.text == "mera naam Rahul hai"  # the turn itself still got recorded
    assert list(ctx.turns) == [turn]
    assert extractor.calls == [turn]
    assert ctx.get_speaker_profile("rahul") is None  # enrichment skipped, not crashed

