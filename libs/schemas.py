"""Common schemas for conversations and data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: Literal["user", "assistant", "system"]
    content: str
    turn_index: int = 0


@dataclass
class Conversation:
    """A complete conversation with multiple turns."""

    turns: List[ConversationTurn] = field(default_factory=list)

    def to_message_list(self):
        """Convert to MessageList format for sampler."""
        return [{"role": turn.role, "content": turn.content} for turn in self.turns]
