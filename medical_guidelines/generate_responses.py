"""Medical guidelines inference - generate conversations from medical guideline questions.

Supports guidelines from any source (NICE, WHO, CDC, etc.) with source-specific prompts.
"""

import asyncio
import json
from pathlib import Path

from libs.conversation_generator import ConversationGenerator
from libs.models import get_sampler
from libs.sampler.openai_sampler import ResponsesSampler
from libs.schemas import Conversation
from libs.storage import save_conversations
from medical_guidelines.data_fetcher import MedicalGuidelineTemplate

from tqdm.asyncio import tqdm


def get_default_instruction(source: str) -> str:
    """Get the default instruction for a specific guideline source.
    
    Args:
        source: The guideline source (e.g., "NICE", "WHO", "CDC")
        
    Returns:
        Instruction string to append to questions
    """
    return (
        # explicitly reference an authoritative guideline
        f"\n\nFor every factual statement, diagnosis, recommendation, or eligibility criterion, immediately follow it with an inline citation.\
                   \n\n - Do not include a References list of uncited items. Every item in References must be cited inline at least once."

    )

async def generate_follow_up_question(
    conversation: Conversation,
    follow_up_sampler: ResponsesSampler,
    guideline_topic: str,
    source: str = "clinical guidelines",
) -> str | None:
    """Generate a follow-up question using LLM based on conversation context.

    Args:
        conversation: Current conversation history
        follow_up_sampler: Sampler for generating follow-up questions
        guideline_topic: Brief description or first line of the guideline being discussed
        source: The guideline source (e.g., "NICE", "WHO", "CDC")

    Returns:
        Follow-up question string, or None if generation fails
    """
    # Format conversation history
    conversation_text = []
    for turn in conversation.turns:
        role = "User" if turn.role == "user" else "Assistant"
        conversation_text.append(f"{role}: {turn.content}")
    conversation_history = "\n".join(conversation_text)

    # Extract first line of guideline as topic
    topic_line = guideline_topic.split('\n')[0][:1000] if guideline_topic else f"{source} guideline"

    prompt = f"""You are a medical education assistant helping to generate natural follow-up questions in a clinical conversation about {source} guidelines.

CONVERSATION CONTEXT:
Guideline topic: {topic_line}
Current conversation:
=== Begin ===
{conversation_history}
=== End ===

TASK: Generate ONE natural, engaging follow-up **question** that a medical student or clinician might ask next. The question should:
1. Build naturally on what has been discussed
2. Show genuine interest in clinical application or guideline details
3. Feel like a natural human question about clinical practice
4. Avoid being too generic or repetitive

Generate only the question text, nothing else:"""

    try:
        message_list = [{"role": "user", "content": prompt}]
        response = await follow_up_sampler(message_list)
        question = response.response_text.strip()

        if not question:
            print("  [X] Failed to generate follow-up question, stopping")
            print("Response:")
            print(response)

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
    """Run inference for medical guidelines task.

    Args:
        data_path: Path to medical guidelines data file (JSONL)
        model_name: Model identifier (e.g., "gpt-5-mini", "gpt-5-mini-high")
        system_prompt_name: Filename of system prompt (e.g., "default.txt")
        output_path: Path to save conversations
        max_concurrent: Max concurrent API calls
        max_follow_ups: Maximum number of follow-up questions (0 = single-turn only)
        follow_up_model_name: Model identifier for generating follow-up questions (None = gpt-5-mini)
        n: Maximum number of conversations to generate (None = use all data)

    Returns:
        Path to saved conversations
    """
    # Simulated user (follow-up questions): default gpt-5-mini unless overridden
    follow_up_model_name = follow_up_model_name or "gpt-5-mini"
    
    # Load system prompt from file
    prompt_path = Path(__file__).parent / "prompts" / system_prompt_name
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        system_message = f.read().strip()

    print(f"Loaded system prompt: {system_prompt_name}\n")

    # Check for existing conversations to continue from
    output_path_obj = Path(output_path)
    output_dir = output_path_obj.parent
    existing_conversations = []
    existing_metadata = []
    latest_file = None
    skip_count = 0
    
    # Look for existing files matching the pattern
    if output_dir.exists():
        pattern = f"conversations_{model_name}_*convs.jsonl"
        existing_files = list(output_dir.glob(pattern))
        if existing_files:
            # Load the most recent file (assuming it has the most conversations)
            from libs.storage import load_conversations
            latest_file = max(existing_files, key=lambda p: p.stat().st_mtime)
            print(f"Found existing conversations file: {latest_file.name}")
            try:
                existing_conversations, existing_metadata = load_conversations(latest_file)
                skip_count = len(existing_conversations)
                print(f"  Loaded {skip_count} existing conversations")
            except Exception as e:
                print(f"  Warning: Could not load existing conversations: {e}")
                existing_conversations = []
                existing_metadata = []
                skip_count = 0

    # Load medical guidelines from data file
    print(f"Loading medical guidelines from {data_path}...")
    guidelines = []
    with open(data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            # Skip the first skip_count items (already processed)
            if idx < skip_count:
                continue
            
            guidelines.append(json.loads(line))
            # Limit the number of NEW guidelines to generate if n is specified
            if n is not None:
                total_needed = n
                already_have = skip_count
                new_needed = total_needed - already_have
                if new_needed <= 0:
                    print(f"[OK] Already have {already_have} conversations (target: {n}), no new conversations needed\n")
                    # Return existing file path
                    if latest_file:
                        return latest_file
                    break
                if len(guidelines) >= new_needed:
                    break
    
    if n is not None:
        total_after = skip_count + len(guidelines)
        print(f"[OK] Loaded {len(guidelines)} new medical guidelines (will have {total_after} total after generation)" + 
              f" (target: {n}, existing: {skip_count})" + "\n")
    else:
        print(f"[OK] Loaded {len(guidelines)} medical guidelines" + 
              (f" (skipped first {skip_count} already processed)" if skip_count > 0 else "") + "\n")

    # Create templates from guidelines
    templates = []
    for guideline in guidelines:
        template = MedicalGuidelineTemplate(
            guideline_text=guideline.get("guideline_text", ""),
            question=guideline.get("question", ""),
            source=guideline.get("source", "clinical guideline"),
        )
        templates.append(template)

    print(f"[OK] Created {len(templates)} templates\n")

    # Create sampler from model registry
    sampler = get_sampler(model_name)
    print(f"Using model: {model_name}\n")

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
        print(f"Using follow-up model: {follow_up_model_name}\n")

    # Generate conversations
    conversation_type = "MULTI-TURN" if max_follow_ups > 0 else "SINGLE-TURN"
    print("=" * 80)
    print(f"GENERATING {len(templates)} {conversation_type} CONVERSATIONS")
    if max_follow_ups > 0:
        print(f"Max follow-ups: {max_follow_ups}")
    print("=" * 80)

    async def generate_conversation_for_guideline(template: MedicalGuidelineTemplate):
        """Generate a conversation (single or multi-turn) for one medical guideline question.
        
        Returns:
            Conversation object, or None if generation failed (e.g., due to safety filters)
        """
        source = template.source
        instruction = get_default_instruction(source)
        
        initial_question = template.question
        initial_question += instruction

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
                        print(f"  [OK] Reached max_follow_ups={max_follow_ups}, stopping")
                        return None

                    next_question = await generate_follow_up_question(
                        conv, follow_up_sampler, 
                        guideline_topic=template.guideline_text,
                        source=source
                    )

                    if next_question:
                        # Append instruction to reference the guideline source
                        next_question += instruction
                        follow_up_count[0] += 1
                        return next_question
                    else:
                        print("  [X] Failed to generate follow-up, stopping")
                        return None

                conversation = await generator.generate_conversation_dynamic(
                    initial_question=initial_question,
                    max_turns=max_turns,
                    follow_up_generator=follow_up_generator,
                )

            return conversation
        except Exception as e:
            error_str = str(e).lower()
            # Check for safety-related API errors
            if "safety" in error_str or "limited access" in error_str or "invalid prompt" in error_str:
                print(f"  [SKIP] Safety filter triggered, skipping this prompt: {str(e)[:100]}")
            else:
                print(f"  [ERROR] Failed to generate conversation: {str(e)[:200]}")
            return None

    # Generate all conversations concurrently
    results = await tqdm.gather(
        *[generate_conversation_for_guideline(template) for template in templates]
    )

    # Filter out None results (failed due to safety filters or other errors)
    successful_pairs = [
        (conv, template) 
        for conv, template in zip(results, templates) 
        if conv is not None
    ]
    
    conversations = [pair[0] for pair in successful_pairs]
    successful_templates = [pair[1] for pair in successful_pairs]
    
    skipped_count = len(results) - len(conversations)
    if skipped_count > 0:
        print(f"\n[!] Skipped {skipped_count} conversations due to errors (safety filters, etc.)")
    
    print(f"\n[OK] Generated {len(conversations)} new conversations successfully\n")

    # Merge with existing conversations if any
    if existing_conversations:
        all_conversations = existing_conversations + conversations
        all_metadata = existing_metadata + [template.to_metadata() for template in successful_templates]
        print(f"  Merging with {len(existing_conversations)} existing conversations")
    else:
        all_conversations = conversations
        all_metadata = [template.to_metadata() for template in successful_templates]
    
    # Update output path to include model name and total conversation count
    from pathlib import Path as PathLib
    output_path_obj = PathLib(output_path)
    total_conversations = len(all_conversations)
    new_filename = f"conversations_{model_name}_{total_conversations}convs.jsonl"
    output_path = str(output_path_obj.parent / new_filename)

    # Save all conversations (overwrite existing file or create new one)
    saved_path = save_conversations(
        conversations=all_conversations,
        output_path=output_path,
        metadata_list=all_metadata,
        task_name="medical_guidelines",
        model_name=model_name,
        system_prompt_name=system_prompt_name,
        append=False,  # Overwrite to create a single merged file
    )

    print(f"\n[OK] Saved {total_conversations} total conversations ({len(conversations)} new) to: {saved_path}")

    return saved_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate medical guideline conversations"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to medical guidelines data file (JSONL)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Model identifier (e.g., gpt-5-mini, gpt-5-mini-high, gpt-5-medium-websearch)",
    )
    parser.add_argument("--system-prompt", type=str, default="default.txt")
    parser.add_argument(
        "--output", type=str, default="medical_guidelines/results/conversations.jsonl"
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
        help="Model for simulated user follow-up questions (default: gpt-5-mini when omitted)",
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
            follow_up_model_name=args.follow_up_model,
            n=args.n,
        )
    )
