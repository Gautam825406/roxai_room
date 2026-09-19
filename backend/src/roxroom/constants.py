"""Tiny cross-module constants -- kept in one place to avoid drift between the
producer (BotAgent, publishing replies) and consumer (TranscriberAgent, reading human
chat messages) of the same LiveKit text-stream topic."""

CHAT_TOPIC = "lk.chat"
