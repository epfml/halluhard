"""Legal cases inference - generate conversations from legal case questions."""

import asyncio
import json
from pathlib import Path

from libs.conversation_generator import ConversationGenerator
from libs.models import get_sampler
from libs.sampler.openai_sampler import ResponsesSampler
from libs.schemas import Conversation
from libs.storage import save_conversations
from legal_cases.data_fetcher import LegalCaseTemplate

from tqdm.asyncio import tqdm


# Default instruction appended to all questions
DEFAULT_INSTRUCTION = "\n\nIf relevant legal cases or law articles exist, cite them precisely (with pinpoint cites) in the answer, not only in References.\
    \n\n - Cite authorities at the point of use: every sentence that relies on a legal authority or external factual source must end with an inline citation. \
    \n\n - Do not include a References list of uncited items. Every item in References must be cited inline at least once."



async def generate_follow_up_question(
    conversation: Conversation,
    follow_up_sampler: ResponsesSampler,
    question_category: str,
) -> str | None:
    """Generate a follow-up question using LLM based on conversation context.

    Args:
        conversation: Current conversation history
        follow_up_sampler: Sampler for generating follow-up questions
        question_category: Category of the legal question being discussed

    Returns:
        Follow-up question string, or None if generation fails
    """
    # Format conversation history
    conversation_text = []
    for turn in conversation.turns:
        role = "User" if turn.role == "user" else "Assistant"
        conversation_text.append(f"{role}: {turn.content}")
    conversation_history = "\n".join(conversation_text)

    prompt = f"""You are a legal assistant helping to generate natural follow-up questions in a conversation about legal cases.

CONVERSATION CONTEXT:
Question Category: {question_category}
Current conversation:
=== Begin ===
{conversation_history}
=== End ===

TASK: Generate ONE natural, engaging follow-up **question** that a legal practitioner or student might ask next. The question should:
1. Build naturally on what has been discussed
2. Show genuine interest in the legal topic
3. Be specific and focused on legal precedents, cases, or principles
4. Feel like a natural human question
5. Avoid being too generic or repetitive

Generate only the question text, nothing else:"""

    try:
        message_list = [{"role": "user", "content": prompt}]
        response = await follow_up_sampler(message_list)
        question = response.response_text.strip()

        return question if question else None
    except Exception as e:
        print(f"  ✗ Error generating follow-up question: {e}")
        return None


async def run_inference(
    data_path: str | Path,
    model_name: str,
    system_prompt_name: str,
    output_path: str | Path,
    max_concurrent: int = 5,
    max_follow_ups: int = 0,
    follow_up_model_name: str | None = None,
    n: int | None = None,
) -> Path:
    """Run inference for legal cases task.

    Args:
        data_path: Path to legal case data file (JSONL)
        model_name: Model identifier (e.g., "gpt-5-mini", "gpt-5-mini-high")
        system_prompt_name: Filename of system prompt (e.g., "default.txt")
        output_path: Path to save conversations
        max_concurrent: Max concurrent API calls
        max_follow_ups: Maximum number of follow-up questions (0 = single-turn only)
        follow_up_model_name: Model identifier for generating follow-up questions (None = use model_name)
        n: Maximum number of conversations to generate (None = use all data)

    Returns:
        Path to saved conversations
    """
    # Default follow-up model to main model if not specified
    follow_up_model_name = follow_up_model_name or model_name
    
    # Load system prompt from file
    prompt_path = Path(__file__).parent / "prompts" / system_prompt_name
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        system_message = f.read().strip()

    print(f"Loaded system prompt: {system_prompt_name}\n")

    # Load legal cases from data file
    print(f"Loading legal cases from {data_path}...")
    cases = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            cases.append(json.loads(line))
            # Limit the number of cases if n is specified
            if n is not None and len(cases) >= n:
                break
    
    if n is not None and len(cases) > n:
        cases = cases[:n]
    
    print(f"✓ Loaded {len(cases)} legal cases" + (f" (limited to {n})" if n is not None else "") + "\n")

    # Create templates from cases
    templates = []
    for case in cases:
        template = LegalCaseTemplate(
            question=case.get("question", ""),
            question_category=case.get("question_category", "Unknown"),
            answer=case.get("answer"),
            source=case.get("source"),
            case_id=case.get("case_id"),
            correctness=case.get("correctness"),
            groundedness=case.get("groundedness"),
            label=case.get("label"),
        )
        templates.append(template)

    print(f"✓ Created {len(templates)} templates\n")

    # Create sampler
    sampler = get_sampler(model_name)

    # Create semaphore for rate limiting
    response_semaphore = asyncio.Semaphore(max_concurrent)
    print(f"Using semaphore: {max_concurrent} concurrent calls\n")

    # Create conversation generator
    generator = ConversationGenerator(
        sampler=sampler,
        system_message=system_message,
        response_semaphore=response_semaphore,
    )

    # Create follow-up question sampler if needed
    follow_up_sampler = None
    if max_follow_ups > 0:
        follow_up_sampler = get_sampler(follow_up_model_name)

    # Generate conversations
    conversation_type = "MULTI-TURN" if max_follow_ups > 0 else "SINGLE-TURN"
    print("=" * 80)
    print(f"GENERATING {len(templates)} {conversation_type} CONVERSATIONS")
    if max_follow_ups > 0:
        print(f"Max follow-ups: {max_follow_ups}")
    print("=" * 80)

    async def generate_conversation_for_case(template: LegalCaseTemplate):
        """Generate a conversation (single or multi-turn) for one legal case question.
        
        Returns:
            Tuple of (conversation, template) or (None, template) if skipped due to content policy.
        """
        initial_question = template.question
        initial_question += DEFAULT_INSTRUCTION

        try:
            if max_follow_ups == 0:
                # Single-turn conversation
                conversation = await generator.generate_conversation(
                    initial_question=initial_question,
                    follow_up_questions=[],
                )
            else:
                # Multi-turn conversation with dynamic follow-ups
                max_turns = (max_follow_ups * 2) + 2  # Initial Q+A + follow-ups
                follow_up_count = [0]  # Use list for mutable closure

                async def follow_up_generator(conv: Conversation, _turn_idx: int):
                    """Generate follow-up questions dynamically."""
                    if follow_up_count[0] >= max_follow_ups:
                        print(f"  ✓ Reached max_follow_ups={max_follow_ups}, stopping")
                        return None

                    next_question = await generate_follow_up_question(
                        conv, follow_up_sampler, question_category=template.question_category
                    )

                    if next_question:
                        # Append instruction to cite legal cases
                        next_question += DEFAULT_INSTRUCTION
                        follow_up_count[0] += 1
                        return next_question
                    else:
                        print("  ✗ Failed to generate follow-up, stopping")
                        return None

                conversation = await generator.generate_conversation_dynamic(
                    initial_question=initial_question,
                    max_turns=max_turns,
                    follow_up_generator=follow_up_generator,
                )

            # Check if the conversation was blocked by content policy
            if conversation.turns and "[CONTENT_POLICY_VIOLATION]" in conversation.turns[0].content:
                print(f"  ⚠ Skipping due to content policy: {template.question[:50]}...")
                return (None, template)
            
            # Also check assistant responses for content policy blocks
            for turn in conversation.turns:
                if turn.role == "assistant" and "[CONTENT_POLICY_VIOLATION]" in turn.content:
                    print(f"  ⚠ Skipping due to content policy: {template.question[:50]}...")
                    return (None, template)

            return (conversation, template)
        except Exception as e:
            print(f"  ✗ Error generating conversation for '{template.question[:50]}...': {e}")
            return (None, template)

    # Generate all conversations concurrently
    results = await tqdm.gather(
        *[generate_conversation_for_case(template) for template in templates]
    )

    # Filter out skipped conversations (where conversation is None)
    valid_results = [(conv, tmpl) for conv, tmpl in results if conv is not None]
    skipped_count = len(results) - len(valid_results)
    
    if skipped_count > 0:
        print(f"\n⚠ Skipped {skipped_count} conversations due to content policy or errors")
    
    conversations = [conv for conv, _ in valid_results]
    valid_templates = [tmpl for _, tmpl in valid_results]

    print(f"\n✓ Generated {len(conversations)} conversations\n")

    # Save conversations with metadata
    metadata_list = [template.to_metadata() for template in valid_templates]

    from pathlib import Path as PathLib
    output_path_obj = PathLib(output_path)
    num_conversations = len(conversations)
    new_filename = f"conversations_{model_name}_{num_conversations}convs.jsonl"
    output_path = str(output_path_obj.parent / new_filename)

    saved_path = save_conversations(
        conversations=conversations,
        output_path=output_path,
        metadata_list=metadata_list,
        task_name="legal_cases",
        model_name=model_name,
        system_prompt_name=system_prompt_name,
    )

    print(f"\n✓ Saved {len(conversations)} conversations to: {saved_path}")

    return saved_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate legal case conversations"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to legal case data file (JSONL)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Model identifier (e.g., gpt-5-mini, gpt-5-mini-high, gpt-5-medium-websearch)",
    )
    parser.add_argument("--system-prompt", type=str, default="default.txt")
    parser.add_argument(
        "--output", type=str, default="legal_cases/results/conversations.jsonl"
    )
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument(
        "--max-follow-ups",
        type=int,
        default=0,
        help="Maximum number of follow-up questions (0 = single-turn only)",
    )
    parser.add_argument(
        "--follow-up-model",
        type=str,
        default=None,
        help="Model identifier for generating follow-up questions (default: same as --model)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Maximum number of conversations to generate (default: None = use all data)",
    )

    args = parser.parse_args()

    asyncio.run(
        run_inference(
            data_path=args.data,
            model_name=args.model,
            system_prompt_name=args.system_prompt,
            output_path=args.output,
            max_concurrent=args.max_concurrent,
            max_follow_ups=args.max_follow_ups,
            follow_up_model_name=args.follow_up_model or args.model,
            n=args.n,
        )
    )

