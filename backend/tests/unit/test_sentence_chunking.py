import pytest

from roxroom.bots.sentence_chunking import sentence_chunk


async def _tokens(pieces: list[str]):
    for p in pieces:
        yield p


@pytest.mark.asyncio
async def test_sentence_chunk_splits_on_sentence_boundaries():
    tokens = ["Ye", "ek", " test", " hai.", " Doosra", " sentence", " hai!", " Teesra?"]
    chunks = [c async for c in sentence_chunk(_tokens(tokens))]
    assert chunks == ["Yeek test hai.", " Doosra sentence hai!", " Teesra?"]


@pytest.mark.asyncio
async def test_sentence_chunk_flushes_trailing_text_without_terminal_punctuation():
    tokens = ["Pehla vaakya.", " Adhoora vaakya bina punctuation ke"]
    chunks = [c async for c in sentence_chunk(_tokens(tokens))]
    assert chunks == ["Pehla vaakya.", " Adhoora vaakya bina punctuation ke"]


@pytest.mark.asyncio
async def test_sentence_chunk_handles_hindi_danda():
    tokens = ["ये एक वाक्य है।", " ये दूसरा है।"]
    chunks = [c async for c in sentence_chunk(_tokens(tokens))]
    assert chunks == ["ये एक वाक्य है।", " ये दूसरा है।"]


@pytest.mark.asyncio
async def test_sentence_chunk_empty_stream_yields_nothing():
    chunks = [c async for c in sentence_chunk(_tokens([]))]
    assert chunks == []
