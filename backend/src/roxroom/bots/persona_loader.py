"""Loads personas/*.yaml into a Persona with a fully composed system prompt: the
shared hard rules (personas/_shared.yaml -- language mixing, banned register, output
format) plus this persona's own traits and few-shot examples (personas/<bot_id>.yaml).

Splitting it this way keeps the persona-*differentiating* content in YAML, as the spec
asks, without duplicating the universal rules across both bot files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PERSONAS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "personas"


@dataclass(frozen=True)
class VoiceConfig:
    provider: str
    voice_id: str


@dataclass(frozen=True)
class Persona:
    bot_id: str
    display_name: str
    gender: str
    voice: VoiceConfig
    system_prompt: str
    degradation_lines: tuple[str, ...] = ()


def _format_few_shot(examples: list[dict]) -> str:
    lines = ["Example exchanges in your voice (match this tone and format, not these exact words):"]
    for ex in examples:
        lines.append(f'Human: "{ex["human"]}"')
        lines.append(f'You: "{ex["bot"]}"')
    return "\n".join(lines)


def load_persona(bot_id: str, personas_dir: Path | str = DEFAULT_PERSONAS_DIR) -> Persona:
    personas_dir = Path(personas_dir)
    shared = yaml.safe_load((personas_dir / "_shared.yaml").read_text(encoding="utf-8"))
    data = yaml.safe_load((personas_dir / f"{bot_id}.yaml").read_text(encoding="utf-8"))

    system_prompt = "\n\n".join(
        part.strip()
        for part in (
            data["persona_description"],
            shared["language_rules"],
            shared["output_rules"],
            shared["barge_in_rules"],
            _format_few_shot(data["few_shot_examples"]),
        )
    )

    return Persona(
        bot_id=data["bot_id"],
        display_name=data["display_name"],
        gender=data["gender"],
        voice=VoiceConfig(provider=data["voice"]["provider"], voice_id=data["voice"]["voice_id"]),
        system_prompt=system_prompt,
        degradation_lines=tuple(data.get("degradation_lines") or ()),
    )
