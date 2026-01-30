"""Generic storage utilities for saving conversations with metadata."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from libs.schemas import Conversation, ConversationTurn


def save_conversations(
    conversations: List[Conversation],
    output_path: str | Path,
    metadata_list: List[Dict[str, Any]] | None = None,
    task_name: str | None = None,
    model_name: str | None = None,
    system_prompt_name: str | None = None,
    append: bool = False,
) -> Path:
    """Save conversations to JSONL file with optional metadata.

    Args:
        conversations: List of Conversation objects to save
        output_path: Path to save the JSONL file (can be relative or absolute)
        metadata_list: Optional list of metadata dicts (one per conversation)
        task_name: Optional task identifier (e.g., "paper_authors")
        model_name: Optional model identifier (e.g., "gpt-5-mini")
        system_prompt_name: Optional filename of system prompt (e.g., "default.txt")
        append: If True, append to existing file; if False, overwrite

    Returns:
        Path object to the saved file

    Example:
        >>> metadata = [
        ...     {"ground_truth_authors": ["Alice", "Bob"], "publication_year": 2023},
        ...     {"ground_truth_authors": ["Carol"], "publication_year": 2024},
        ... ]
        >>> path = save_conversations(
        ...     conversations=conversations,
        ...     output_path="paper_authors/results/conversations.jsonl",
        ...     metadata_list=metadata,
        ...     task_name="paper_authors",
        ...     model_name="gpt-5-mini",
        ... )
    """
    # Ensure output_path is a Path object
    output_path = Path(output_path)

    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate metadata_list length if provided
    if metadata_list and len(metadata_list) != len(conversations):
        raise ValueError(
            f"metadata_list length ({len(metadata_list)}) must match "
            f"conversations length ({len(conversations)})"
        )

    # Determine write mode
    mode = "a" if append else "w"

    # Write conversations as JSONL
    with open(output_path, mode, encoding="utf-8") as f:
        for idx, conversation in enumerate(conversations):
            # Build the record
            record = {
                "conversation_id": idx,
                "timestamp": datetime.now().isoformat(),
                "turns": [
                    {
                        "role": turn.role,
                        "content": turn.content,
                        "turn_index": turn.turn_index,
                    }
                    for turn in conversation.turns
                ],
            }

            # Add optional fields
            if task_name:
                record["task_name"] = task_name
            if model_name:
                record["model_name"] = model_name
            if system_prompt_name:
                record["system_prompt_name"] = system_prompt_name

            # Add metadata if provided
            if metadata_list:
                record["metadata"] = metadata_list[idx]

            # Write as JSON line (with readable UTF-8 characters)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def load_conversations(
    input_path: str | Path,
) -> tuple[List[Conversation], List[Dict[str, Any]]]:
    """Load conversations from JSONL file.

    Args:
        input_path: Path to the JSONL file

    Returns:
        Tuple of (conversations, metadata_list)

    Example:
        >>> conversations, metadata = load_conversations("results/paper_authors_results.jsonl")
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    conversations = []
    metadata_list = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            # Reconstruct Conversation object
            turns = [
                ConversationTurn(
                    role=turn["role"],
                    content=turn["content"],
                    turn_index=turn["turn_index"],
                )
                for turn in record["turns"]
            ]

            conversation = Conversation(turns=turns)
            conversations.append(conversation)

            # Extract metadata if present
            metadata = record.get("metadata", {})
            # Add task, model, system_prompt_name, and other inference info to metadata
            if "task_name" in record:
                metadata["task_name"] = record["task_name"]
            if "model_name" in record:
                metadata["model_name"] = record["model_name"]
            if "system_prompt_name" in record:
                metadata["system_prompt_name"] = record["system_prompt_name"]
            if "conversation_id" in record:
                metadata["conversation_id"] = record["conversation_id"]
            if "timestamp" in record:
                metadata["timestamp"] = record["timestamp"]

            metadata_list.append(metadata)

    return conversations, metadata_list


def save_conversation_single(
    conversation: Conversation,
    output_path: str | Path,
    metadata: Dict[str, Any] | None = None,
    task_name: str | None = None,
    model_name: str | None = None,
    append: bool = True,
) -> Path:
    """Save a single conversation (convenience wrapper).

    Args:
        conversation: Single Conversation object to save
        output_path: Path to save the JSONL file
        metadata: Optional metadata dict
        task_name: Optional task identifier
        model_name: Optional model identifier
        append: If True, append to existing file; if False, overwrite

    Returns:
        Path object to the saved file
    """
    return save_conversations(
        conversations=[conversation],
        output_path=output_path,
        metadata_list=[metadata] if metadata else None,
        task_name=task_name,
        model_name=model_name,
        append=append,
    )
