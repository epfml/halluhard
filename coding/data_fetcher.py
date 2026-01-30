"""Coding task data fetcher - generates coding prompts from tasks and templates.

This module:
1. Loads tasks and prompt templates from data files
2. Validates task-language combinations using LLM (optional)
3. Generates prompts for each (language, task, template) combination
4. Saves prompts to JSONL format for inference
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI

import dotenv
dotenv.load_dotenv()


@dataclass
class CodingPromptTemplate:
    """Template for coding task prompts with metadata."""

    language: str
    task: str
    prompt_template: str
    prompt: str

    def to_metadata(self) -> Dict[str, Any]:
        """Convert template to metadata dict for storage."""
        return {
            "language": self.language,
            "task": self.task,
            "prompt_template": self.prompt_template,
            "prompt": self.prompt,
        }


def load_tasks(task_file: Path) -> list[str]:
    """Load tasks from file, ignoring comments and empty lines.

    Args:
        task_file: Path to tasks file

    Returns:
        List of task strings
    """
    tasks = []
    with open(task_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            tasks.append(line)
    return tasks


def load_prompt_templates(prompt_file: Path) -> list[str]:
    """Load prompt templates from file, ignoring comments and empty lines.

    Args:
        prompt_file: Path to prompt templates file

    Returns:
        List of prompt template strings
    """
    templates = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            templates.append(line)
    return templates


def validate_and_suggest_alternative(
    task: str, language: str, client: OpenAI
) -> tuple[bool, str | None]:
    """Validate if a task makes sense for a language, and suggest alternative if not.

    Args:
        task: The coding task description
        language: The programming language
        client: OpenAI client for API calls

    Returns:
        Tuple of (is_valid, alternative_task)
        - is_valid: True if the combination is valid, False otherwise
        - alternative_task: A valid alternative task for this language if invalid, None otherwise
    """
    validation_prompt = f"""You are a programming language expert. Determine if the following task makes sense for the given programming language.

Task: {task}
Language: {language}

Consider:
- Is this task technically feasible in this language?
- Are the required libraries/frameworks typically available in this language?
- Is this language commonly used for this type of task?

Examples of INVALID combinations:
- "Interface with a Flask API" in R (Flask is Python-specific)
- "Load a CUDA kernel" in R (CUDA is typically C/C++/Python)
- "Embed a Rust library in Python" in Scala (task is Python-specific)
- "Run inference on a Coral TPU" in R (TPU inference typically Python/C++)

If the combination is VALID, respond with: "VALID"

If the combination is INVALID, respond with: "INVALID: [suggest a similar but valid task for {language}]"

Example responses:
- "VALID"
- "INVALID: Load a statistical computing package for data analysis"
- "INVALID: Interface with a RESTful API using HTTP client libraries"

Keep suggestions concise and similar in complexity to the original task.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a programming language expert. Validate tasks and suggest alternatives.",
                },
                {"role": "user", "content": validation_prompt},
            ],
            temperature=0.3,  # Slight creativity for suggestions
            max_tokens=100,
        )

        answer = response.choices[0].message.content.strip()
        
        if answer.upper().startswith("VALID"):
            return True, None
        elif answer.upper().startswith("INVALID:"):
            # Extract the suggested alternative
            alternative = answer.split(":", 1)[1].strip()
            return False, alternative
        else:
            # Fallback: couldn't parse response, accept the combination
            print(f"Warning: Unexpected response for '{task}' in {language}: {answer}")
            return True, None

    except Exception as e:
        print(f"Warning: Validation failed for '{task}' in {language}: {e}")
        # On error, default to accepting the combination
        return True, None


def filter_valid_tasks(
    languages: list[str], tasks: list[str], use_llm_validation: bool
) -> dict[str, list[str]]:
    """Filter tasks for each language, replacing invalid ones with LLM suggestions.

    Args:
        languages: List of programming languages
        tasks: List of coding tasks
        use_llm_validation: Whether to use LLM for validation

    Returns:
        Dictionary mapping each language to its list of valid tasks
    """
    if not use_llm_validation:
        print("Skipping LLM validation (--no-validate flag used)")
        # Return all tasks for all languages
        return {lang: tasks for lang in languages}

    print("\nValidating task-language combinations with LLM...")
    print("(This may take a few minutes)")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    valid_tasks_per_language = {}

    for lang in languages:
        print(f"\nValidating {lang}...")
        valid_tasks = []
        replaced_count = 0

        for i, task in enumerate(tasks, 1):
            is_valid, alternative = validate_and_suggest_alternative(task, lang, client)
            
            if is_valid:
                valid_tasks.append(task)
            elif alternative:
                # Replace with suggested alternative
                valid_tasks.append(alternative)
                replaced_count += 1
                if i <= 10 or i % 10 == 0:  # Show early replacements and periodic updates
                    print(f"  ✎ Replaced: '{task[:50]}...' → '{alternative[:50]}...'")
            # If neither valid nor has alternative, skip the task

            # Print progress every 10 tasks
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(tasks)} tasks processed")

        valid_tasks_per_language[lang] = valid_tasks
        kept_count = len(valid_tasks) - replaced_count
        print(
            f"  [OK] {lang}: {kept_count} kept, {replaced_count} replaced, "
            f"{len(tasks) - len(valid_tasks)} skipped → {len(valid_tasks)} total tasks"
        )

    return valid_tasks_per_language


def generate_prompts(
    languages: list[str],
    valid_tasks_per_language: dict[str, list[str]],
    prompt_templates: list[str],
    samples_per_language: int,
    random_seed: int,
) -> list[CodingPromptTemplate]:
    """Generate coding prompts from tasks and templates.

    Args:
        languages: List of programming languages
        valid_tasks_per_language: Dictionary mapping language to valid tasks
        prompt_templates: List of prompt template strings
        samples_per_language: Number of samples to generate per language
        random_seed: Random seed for reproducibility

    Returns:
        List of CodingPromptTemplate objects
    """
    random.seed(random_seed)
    templates = []
    total_sampled = 0

    # Generate prompts for each language using its valid tasks
    for lang in languages:
        lang_tasks = valid_tasks_per_language[lang]

        if not lang_tasks:
            print(f"Warning: No valid tasks for {lang}, skipping")
            continue

        # Build all (task, template) pairs for this language
        all_pairs = [(t, p) for t, p in product(lang_tasks, prompt_templates)]

        if not all_pairs:
            print(f"Warning: No (task, template) pairs for {lang}, skipping")
            continue

        # Sample a subset of pairs for this language
        k = min(samples_per_language, len(all_pairs))
        sampled_pairs = random.sample(all_pairs, k)
        total_sampled += k

        print(
            f"  {lang}: Sampled {k} pairs from {len(all_pairs)} possible combinations"
        )

        # Generate prompts for this language
        for task, prompt_tmpl in sampled_pairs:
            # Interpolate template with language and task
            prompt = prompt_tmpl.format(language=lang, task=task)

            templates.append(
                CodingPromptTemplate(
                    language=lang,
                    task=task,
                    prompt_template=prompt_tmpl,
                    prompt=prompt,
                )
            )

    print(f"\nTotal prompts generated: {len(templates)}")
    return templates


def save_prompts(templates: list[CodingPromptTemplate], output_path: Path) -> None:
    """Save prompts to JSONL file.

    Args:
        templates: List of CodingPromptTemplate objects
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for template in templates:
            f.write(json.dumps(template.to_metadata(), ensure_ascii=False) + "\n")

    print(f"\n[OK] Saved {len(templates)} prompts to: {output_path}")


def main_pipeline(
    languages: list[str],
    samples_per_language: int,
    random_seed: int,
    task_file: Path,
    prompt_file: Path,
    output_path: Path,
    use_llm_validation: bool = True,
) -> None:
    """Main pipeline to generate coding prompts.

    Args:
        languages: List of programming languages
        samples_per_language: Number of samples per language
        random_seed: Random seed for reproducibility
        task_file: Path to tasks file
        prompt_file: Path to prompt templates file
        output_path: Path to output file
        use_llm_validation: Whether to validate task-language combinations with LLM
    """
    print("=" * 80)
    print("CODING TASK DATA FETCHER")
    print("=" * 80)

    # Load tasks and templates
    print(f"\nLoading tasks from {task_file}...")
    tasks = load_tasks(task_file)
    print(f"[OK] Loaded {len(tasks)} tasks")

    print(f"\nLoading prompt templates from {prompt_file}...")
    prompt_templates = load_prompt_templates(prompt_file)
    print(f"[OK] Loaded {len(prompt_templates)} prompt templates")

    print(f"\nLanguages: {', '.join(languages)}")

    # Validate task-language combinations
    valid_tasks_per_language = filter_valid_tasks(languages, tasks, use_llm_validation)

    # Generate prompts
    print("\nGenerating prompts...")
    templates = generate_prompts(
        languages,
        valid_tasks_per_language,
        prompt_templates,
        samples_per_language,
        random_seed,
    )

    # Save to file
    save_prompts(templates, output_path)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Languages: {len(languages)}")
    print(f"Total tasks loaded: {len(tasks)}")
    if use_llm_validation:
        print("\nTask validation results:")
        for lang in languages:
            task_count = len(valid_tasks_per_language[lang])
            print(f"  {lang}: {task_count} tasks available after validation")
    print(f"\nPrompt templates: {len(prompt_templates)}")
    print(f"Samples per language: {samples_per_language}")
    print(f"Total prompts generated: {len(templates)}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate coding prompts from tasks and templates"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["Python", "Scala", "Elixir", "R"],
        help="Programming languages to generate prompts for",
    )
    parser.add_argument(
        "--samples-per-language",
        type=int,
        default=20,
        help="Number of (task, template) samples per language",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--task-file",
        type=str,
        default=None,
        help="Path to tasks file (default: coding/data/tasks.txt)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to prompt templates file (default: coding/data/prompt_templates.txt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: coding/data/coding_prompts.jsonl)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip LLM validation of task-language combinations (faster but may generate nonsensical prompts)",
    )

    args = parser.parse_args()

    # Set defaults relative to script location
    script_dir = Path(__file__).parent
    task_file = (
        Path(args.task_file) if args.task_file else script_dir / "data" / "tasks.txt"
    )
    prompt_file = (
        Path(args.prompt_file)
        if args.prompt_file
        else script_dir / "data" / "prompt_templates.txt"
    )
    output_file = (
        Path(args.output)
        if args.output
        else script_dir / "data" / "coding_prompts.jsonl"
    )

    # Validate input files exist
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt template file not found: {prompt_file}")

    # Run pipeline
    main_pipeline(
        languages=args.languages,
        samples_per_language=args.samples_per_language,
        random_seed=args.seed,
        task_file=task_file,
        prompt_file=prompt_file,
        output_path=output_file,
        use_llm_validation=not args.no_validate,
    )

