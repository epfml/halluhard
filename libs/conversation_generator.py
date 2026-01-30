"""Generic multi-turn conversation generator."""

from __future__ import annotations

import asyncio
from typing import List

from libs.schemas import Conversation, ConversationTurn
from libs.types import SamplerBase

from tqdm.asyncio import tqdm


class ConversationGenerator:
    """Generic conversation generator that uses a sampler to generate responses.

    This is domain-agnostic and can be used for any type of conversation.
    Domain-specific question generation should be handled by wrappers.

    Supports concurrency control via semaphores to respect API rate limits.
    """

    def __init__(
        self,
        sampler: SamplerBase,
        system_message: str | None = None,
        response_semaphore: asyncio.Semaphore | None = None,
    ):
        """Initialize the conversation generator.

        Args:
            sampler: SamplerBase instance for generating responses (required)
            system_message: Optional system message to prepend to conversations
            response_semaphore: Optional semaphore to control concurrent response generation
                              (helps prevent API rate limiting)
        """
        self.sampler = sampler
        self.system_message = system_message
        self.response_semaphore = response_semaphore

    async def generate_conversation(
        self,
        initial_question: str,
        follow_up_questions: List[str] | None = None,
    ) -> Conversation:
        """Generate a conversation given initial and follow-up questions.

        Args:
            initial_question: The first user question
            follow_up_questions: Optional list of follow-up questions

        Returns:
            Conversation object with all turns
        """
        turns = []
        follow_up_questions = follow_up_questions or []

        # Add initial question
        turns.append(
            ConversationTurn(role="user", content=initial_question, turn_index=0)
        )

        # Generate response to initial question
        message_list = self._build_message_list(turns)
        response = await self._get_response(message_list)

        turns.append(ConversationTurn(role="assistant", content=response, turn_index=1))

        # Generate follow-up turns
        current_turn = 2
        for follow_up_q in follow_up_questions:
            # Add follow-up question
            turns.append(
                ConversationTurn(
                    role="user", content=follow_up_q, turn_index=current_turn
                )
            )
            current_turn += 1

            # Generate response
            message_list = self._build_message_list(turns)
            response = await self._get_response(message_list)

            turns.append(
                ConversationTurn(
                    role="assistant", content=response, turn_index=current_turn
                )
            )
            current_turn += 1

        return Conversation(turns=turns)

    async def generate_conversation_dynamic(
        self,
        initial_question: str,
        max_turns: int = 6,
        follow_up_generator=None,
    ) -> Conversation:
        """Generate a conversation with dynamically generated follow-ups.

        Args:
            initial_question: The first user question
            max_turns: Maximum number of turns
            follow_up_generator: Async callable that takes (conversation, turn_index)
                                and returns next question or None

        Returns:
            Conversation object with all turns
        """
        turns = []

        # Add initial question
        turns.append(
            ConversationTurn(role="user", content=initial_question, turn_index=0)
        )

        current_turn = 1

        while current_turn < max_turns:
            # Generate assistant response
            message_list = self._build_message_list(turns)
            response = await self._get_response(message_list)

            turns.append(
                ConversationTurn(
                    role="assistant", content=response, turn_index=current_turn
                )
            )
            current_turn += 1

            # Generate follow-up question if generator provided and within limits
            if current_turn < max_turns and follow_up_generator is not None:
                next_question = await follow_up_generator(
                    Conversation(turns=turns), current_turn
                )

                if next_question:
                    turns.append(
                        ConversationTurn(
                            role="user", content=next_question, turn_index=current_turn
                        )
                    )
                    current_turn += 1
                else:
                    # No more follow-ups
                    break
            else:
                # No generator or reached max turns
                break

        return Conversation(turns=turns)

    async def _get_response(self, message_list: List[dict]) -> str:
        """Get a response from the sampler, respecting semaphore limits."""
        if self.response_semaphore:
            async with self.response_semaphore:
                sampler_response = await self.sampler(message_list)
                return sampler_response.response_text
        else:
            sampler_response = await self.sampler(message_list)
            return sampler_response.response_text

    def _build_message_list(self, turns: List[ConversationTurn]) -> List[dict]:
        """Build message list for sampler, optionally prepending system message."""
        messages = []

        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})

        for turn in turns:
            messages.append({"role": turn.role, "content": turn.content})

        return messages

    async def generate_dataset(
        self,
        questions_with_followups: List[tuple[str, List[str]]],
    ) -> List[Conversation]:
        """Generate a dataset of conversations in parallel.

        Args:
            questions_with_followups: List of (initial_question, follow_ups) tuples

        Returns:
            List of Conversation objects
        """

        conversations = await tqdm.gather(
            *[
                self.generate_conversation(initial_q, follow_ups)
                for initial_q, follow_ups in questions_with_followups
            ]
        )

        return conversations
