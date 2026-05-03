"""Coding task inference - generate code responses from coding prompts."""

import asyncio
import json
from pathlib import Path

from libs.conversation_generator import ConversationGenerator
from libs.models import get_sampler
from libs.sampler.openai_sampler import ResponsesSampler
from libs.schemas import Conversation
from libs.storage import save_conversations
from coding.data_fetcher import CodingPromptTemplate

from tqdm.asyncio import tqdm

import dotenv
dotenv.load_dotenv()

# Default instruction appended to all questions
DEFAULT_INSTRUCTION = "\n\nPlease provide the code solution along with installation instructions for the required libraries."


async def generate_follow_up_question(
    conversation: Conversation,
    follow_up_sampler: ResponsesSampler,
    language: str,
    task: str,
) -> str | None:
    """Generate a follow-up question using LLM based on conversation context.

    Args:
        conversation: Current conversation history
        follow_up_sampler: Sampler for generating follow-up questions
        language: Programming language being discussed
        task: Coding task being discussed

    Returns:
        Follow-up question string, or None if generation fails
    """
    # Format conversation history
    conversation_text = []
    for turn in conversation.turns:
        role = "User" if turn.role == "user" else "Assistant"
        conversation_text.append(f"{role}: {turn.content}")
    conversation_history = "\n".join(conversation_text)

    prompt = f"""You are a coding assistant helping to generate natural follow-up questions in a conversation about programming.

CONVERSATION CONTEXT:
Language: {language}
Task: {task}
Current conversation:
=== Begin ===
{conversation_history}
=== End ===

TASK: Generate ONE natural, engaging follow-up **question** that a curious developer might ask next. The question should:
1. Build naturally on what has been discussed in the code
2. Show genuine interest in the implementation details, edge cases, or improvements
3. Be specific and focused on the {language} code
4. Feel like a natural human question (e.g., about testing, optimization, error handling, alternatives)
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
    """Run inference for coding task.

    Args:
        data_path: Path to coding prompt data file (JSONL)
        model_name: Model identifier (e.g., "gpt-5-mini", "gpt-5-mini-high")
        system_prompt_name: Filename of system prompt (e.g., "default.txt")
        output_path: Path to save conversations
        max_concurrent: Max concurrent API calls
        max_follow_ups: Maximum number of follow-up questions (0 = single-turn only)
        follow_up_model_name: Model identifier for generating follow-up questions (None = gpt-5-mini)
        n: Maximum number of prompts per language to use (None = use all data)

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

    # Check for existing conversations
    output_path_obj = Path(output_path)
    output_dir = output_path_obj.parent
    existing_conversations = []
    existing_metadata = []
    latest_file = None
    counts_per_language = {}
    
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
                # Count how many conversations per language have been processed
                for metadata in existing_metadata:
                    lang = metadata.get("language", "unknown")
                    counts_per_language[lang] = counts_per_language.get(lang, 0) + 1
                total_existing = len(existing_conversations)
                print(f"  Loaded {total_existing} existing conversations")
                if counts_per_language:
                    lang_summary = ", ".join([f"{lang}: {count}" for lang, count in sorted(counts_per_language.items())])
                    print(f"  Per language: {lang_summary}")
            except Exception as e:
                print(f"  Warning: Could not load existing conversations: {e}")
                existing_conversations = []
                existing_metadata = []
                counts_per_language = {}

    # Load coding prompts from data file
    print(f"Loading coding prompts from {data_path}...")
    all_prompts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            all_prompts.append(json.loads(line))

    # Group prompts by language
    prompts_by_language = {}
    for prompt in all_prompts:
        lang = prompt.get("language", "unknown")
        if lang not in prompts_by_language:
            prompts_by_language[lang] = []
        prompts_by_language[lang].append(prompt)

    # If n is specified, limit to n prompts per language (accounting for existing)
    if n is not None:
        prompts = []
        all_languages_have_enough = True
        
        for lang, lang_prompts in sorted(prompts_by_language.items()):
            already_have = counts_per_language.get(lang, 0)
            new_needed = n - already_have
            
            if new_needed <= 0:
                # Already have enough for this language
                print(f"  {lang}: Already have {already_have} (target: {n}), skipping")
                continue
            else:
                all_languages_have_enough = False
                # Take the next new_needed prompts (skip the first already_have)
                prompts_to_add = lang_prompts[already_have:already_have + new_needed]
                prompts.extend(prompts_to_add)
                print(f"  {lang}: Need {new_needed} more (have {already_have}, target: {n})")
        
        if all_languages_have_enough and existing_conversations:
            print(f"\n✓ Already have {len(existing_conversations)} conversations (target: {n} per language), no new conversations needed\n")
            # Return existing file path
            if latest_file:
                return latest_file
        
        total_after = len(existing_conversations) + len(prompts)
        print(
            f"\n[OK] Loaded {len(prompts)} new coding prompts (will have {total_after} total after generation)"
            f" ({n} per language from {len(prompts_by_language)} languages)\n"
        )
    else:
        # If no n specified, skip already processed prompts
        if counts_per_language:
            prompts = []
            for lang, lang_prompts in sorted(prompts_by_language.items()):
                already_have = counts_per_language.get(lang, 0)
                # Skip the first already_have prompts for this language
                prompts.extend(lang_prompts[already_have:])
            print(
                f"[OK] Loaded {len(prompts)} coding prompts"
                f" (skipped {len(existing_conversations)} already processed)\n"
            )
        else:
            prompts = all_prompts
            print(f"[OK] Loaded {len(prompts)} coding prompts\n")

    # If no new prompts to process, return early
    if not prompts:
        if latest_file:
            print(f"\n✓ No new prompts to process, returning existing file: {latest_file}\n")
            return latest_file
        else:
            raise ValueError("No prompts to process and no existing conversations found")

    # Create templates from prompts
    templates = []
    for prompt_data in prompts:
        template = CodingPromptTemplate(
            language=prompt_data["language"],
            task=prompt_data["task"],
            prompt_template=prompt_data["prompt_template"],
            prompt=prompt_data["prompt"],
        )
        templates.append(template)

    print(f"[OK] Created {len(templates)} templates\n")

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

    async def generate_conversation_for_prompt(template: CodingPromptTemplate):
        """Generate a conversation (single or multi-turn) for one coding prompt."""
        try:
            initial_question = template.prompt + DEFAULT_INSTRUCTION

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
                        return None

                    next_question = await generate_follow_up_question(
                        conv, follow_up_sampler, language=template.language, task=template.task
                    )

                    if next_question:
                        # Append installation instructions reminder to follow-up
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

            return (conversation, template)
        except Exception as e:
            print(f"  ✗ Error generating conversation for '{template.task[:50]}...': {e}")
            return (None, template)

    # Generate all conversations concurrently
    results = await tqdm.gather(
        *[generate_conversation_for_prompt(template) for template in templates]
    )

    # Filter out failed conversations
    valid_results = [(conv, tmpl) for conv, tmpl in results if conv is not None]
    skipped_count = len(results) - len(valid_results)

    if skipped_count > 0:
        print(f"\n⚠ Skipped {skipped_count} conversations due to errors")

    conversations = [conv for conv, _ in valid_results]
    valid_templates = [tmpl for _, tmpl in valid_results]

    print(f"\n[OK] Generated {len(conversations)} conversations\n")

    # Merge with existing conversations if any
    if existing_conversations:
        all_conversations = existing_conversations + conversations
        all_metadata = existing_metadata + [template.to_metadata() for template in valid_templates]
        print(f"  Merging with {len(existing_conversations)} existing conversations")
    else:
        all_conversations = conversations
        all_metadata = [template.to_metadata() for template in valid_templates]

    # Update output path to include model name and conversation count
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
        task_name="coding",
        model_name=model_name,
        system_prompt_name=system_prompt_name,
        append=False,  # Overwrite to create a single merged file
    )

    print(f"\n[OK] Saved {total_conversations} total conversations ({len(conversations)} new) to: {saved_path}")

    return saved_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate coding task conversations")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to coding prompt data file (JSONL)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Model identifier (e.g., gpt-5-mini, gpt-5-mini-high, gpt-5-medium-websearch)",
    )
    parser.add_argument("--system-prompt", type=str, default="default.txt")
    parser.add_argument(
        "--output", type=str, default="coding/results/conversations.jsonl"
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
        help="Maximum number of prompts per language (default: None = use all data)",
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

