"""Pre-TTS text normalization: the Hinglish script-mixing problem + acronym readability.

DECISION (full writeup in DECISIONS.md): we use *both* options the spec poses, not
either alone, because they cover different failure modes.

Primary: the persona system prompts (personas/*.yaml) instruct the LLM to emit Hindi
words in Devanagari and English/loanwords in Latin within the same sentence. The LLM
has full sentence context, so it can tell "main" (Hindi "I") from "main" (English, as
in "main point") in a way a standalone script pass never could -- that ambiguity is
exactly why we don't try to do this with a blanket transliterator (see below).

Secondary / safety net: this module, applied to the LLM's output right before TTS.

We tried a generic phonetic transliterator first -- `indic-transliteration`'s ITRANS
engine -- and measured it against real words instead of assuming it would work. It
produces WRONG Devanagari for casually-typed Hinglish, because ITRANS expects strict
long-vowel/doubled-consonant marking that casual typing doesn't follow:
    kya    -> क्य   (wrong; should be क्या)
    accha  -> अच्च  (wrong; should be अच्छा)
    batao  -> बतओ   (wrong; should be बताओ)
Shipping that would make pronunciation worse, not better, so we rejected it.

What we ship instead: a small, hand-verified lexicon covering only high-frequency
discourse/question words that are (a) the ones most butchered by English-phonetic TTS
reading -- the spec's own example is "kya" -> "kai-ya" -- and (b) have zero realistic
English-homograph collision. We deliberately EXCLUDE words like "do" (Hindi "two" /
English verb "do"), "is" (Hindi oblique "this" / English "is"), "main" (Hindi "I" /
English adjective "main"), and "the" (Hindi past-tense "were" / English article) --
converting the wrong sense of one of these would corrupt an English sentence. Better to
leave a genuinely ambiguous word in Latin than mangle it; the primary strategy (prompt
instruction) is what's supposed to handle those correctly using sentence context.
"""
from __future__ import annotations

import re

HINGLISH_LEXICON: dict[str, str] = {
    "kya": "क्या", "kaise": "कैसे", "kaisa": "कैसा", "kaisi": "कैसी",
    "kyun": "क्यों", "kyu": "क्यों", "kyunki": "क्योंकि",
    "matlab": "मतलब", "accha": "अच्छा", "achha": "अच्छा", "acha": "अच्छा",
    "nahi": "नहीं", "nahin": "नहीं", "haan": "हाँ", "haanji": "हाँजी",
    "theek": "ठीक", "thik": "ठीक", "bilkul": "बिल्कुल",
    "shayad": "शायद", "zaroor": "ज़रूर", "zarur": "ज़रूर",
    "lekin": "लेकिन", "magar": "मगर",
    "samjho": "समझो", "samjhao": "समझाओ", "samjhana": "समझाना",
    "batao": "बताओ", "bataana": "बताना", "batana": "बताना",
    "bolo": "बोलो", "suno": "सुनो", "dekho": "देखो", "chalo": "चलो",
    "ruk": "रुक", "ruko": "रुको",
    "tumhe": "तुम्हें", "tumhara": "तुम्हारा", "tumhari": "तुम्हारी",
    "mujhe": "मुझे", "aapko": "आपको", "aapka": "आपका", "aapki": "आपकी",
    "humein": "हमें", "humara": "हमारा",
    "uska": "उसका", "uski": "उसकी", "iska": "इसका", "iski": "इसकी",
    "pareshan": "परेशान", "samajh": "समझ",
    "sawaal": "सवाल", "jawab": "जवाब", "waise": "वैसे",
    "abhi": "अभी", "subah": "सुबह", "shaam": "शाम",
    "hai": "है", "hain": "हैं", "tha": "था", "thi": "थी",
}

# Deliberately NOT in the lexicon (English-homograph collisions -- see module docstring):
# do, is, us, so, to, main, the, ho.

ACRONYM_MAP: dict[str, str] = {
    "AI": "ए आई",
    "IPL": "आई पी एल",
    "5G": "फाइव जी",
    "4G": "फोर जी",
    "UPI": "यू पी आई",
    "OTP": "ओ टी पी",
}

# Full Hindi number-to-words (e.g. "2024" -> "दो हज़ार चौबीस") is a known gap: no
# reliable library exists (num2words has no Hindi support) and getting Hindi's
# irregular 1-99 forms right without native-speaker verification is easy to get wrong.
# Bare digit sequences pass through unchanged; most Hindi/multilingual TTS voices
# already read digits reasonably well in context.

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]|\s+")


def normalize_for_tts(text: str) -> str:
    """Acronym map first (case-sensitive), then the Hinglish lexicon (case-insensitive),
    word by word. Punctuation and spacing are preserved exactly. Anything not
    recognized -- including every English/loanword -- passes through unchanged."""

    def replace(match: re.Match) -> str:
        token = match.group()
        if token in ACRONYM_MAP:
            return ACRONYM_MAP[token]
        return HINGLISH_LEXICON.get(token.lower(), token)

    return _TOKEN_RE.sub(replace, text)
