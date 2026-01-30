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
        follow_up_model_name: Model identifier for generating follow-up questions (None = use model_name)
        n: Maximum number of prompts per language to use (None = use all data)

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

    # Load coding prompts from data file
    print(f"Loading coding prompts from {data_path}...")
    all_prompts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            all_prompts.append(json.loads(line))

    # If n is specified, limit to n prompts per language
    if n is not None:
        prompts_by_language = {}
        for prompt in all_prompts:
            lang = prompt.get("language", "unknown")
            if lang not in prompts_by_language:
                prompts_by_language[lang] = []
            prompts_by_language[lang].append(prompt)
        
        # Take first n from each language
        prompts = []
        for lang, lang_prompts in sorted(prompts_by_language.items()):
            prompts.extend(lang_prompts[:n])
        
        print(
            f"[OK] Loaded {len(prompts)} coding prompts ({n} per language from {len(prompts_by_language)} languages)\n"
        )
    else:
        prompts = all_prompts
        print(f"[OK] Loaded {len(prompts)} coding prompts\n")

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

        return conversation

    # Generate all conversations concurrently
    conversations = await tqdm.gather(
        *[generate_conversation_for_prompt(template) for template in templates]
    )

    print(f"\n[OK] Generated {len(conversations)} conversations\n")

    # Save conversations with metadata
    metadata_list = [template.to_metadata() for template in templates]
    
    # Update output path to include model name and conversation count
    from pathlib import Path as PathLib
    output_path_obj = PathLib(output_path)
    num_conversations = len(conversations)
    new_filename = f"conversations_{model_name}_{num_conversations}convs.jsonl"
    output_path = str(output_path_obj.parent / new_filename)

    saved_path = save_conversations(
        conversations=conversations,
        output_path=output_path,
        metadata_list=metadata_list,
        task_name="coding",
        model_name=model_name,
        system_prompt_name=system_prompt_name,
    )

    print(f"\n[OK] Saved {len(conversations)} conversations to: {saved_path}")

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
        help="Model identifier for generating follow-up questions (default: same as --model)",
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
            follow_up_model_name=args.follow_up_model or args.model,
            n=args.n,
        )
    )

