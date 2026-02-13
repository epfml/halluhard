# HalluHard: A Hard Multi-Turn Hallucination Benchmark

A framework for evaluating hallucinations in multi-turn conversations across challenging domains.

Check out our website: https://halluhard.com/

[Update: Feb 12, 2026] We’ve added the following models to our leaderboard: Kimi-K2.5-thinking (released Jan 29, 2026), GLM-5-thinking (released Feb 11, 2026), Grok-4-thinking, and Grok-4.1-thinking-fast.

## Preparation

We use pixi to make sure our experiments are reproducible across all enviroments.

### Installation

Install [pixi](https://pixi.sh) package manager:

**Linux/macOS:**
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

**Windows:**
Download the installer from [pixi.sh](https://pixi.sh/latest/installation/#__tabbed_1_2) and run it.

### Configuration

Running models from hosted providers typically requires API credentials. Export the relevant environment variables for the providers/models you plan to use.

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
# Add others as needed (e.g., Google/DeepSeek/Moonshot), depending on what you run.
```


## Quick Start

### 1. Generate Responses

Generate multi-turn conversations using different model:

```bash
# Example: Research Questions task
pixi run python -m research_questions.generate_responses \
  --data research_questions/data/research_questions_all.jsonl \
  --model gpt-5 \
  --max-follow-ups 2 \
  --max-concurrent 100 \
```

Tips: 
- You may need to change `--max-concurrent` to a smaller value if a rate limit is reached. 
- Start with a tiny run first (the example below generates 3 conversations, each with 2 follow-up questions):

```bash
# Example: Research Questions task
pixi run python -m research_questions.generate_responses \
  --data research_questions/data/research_questions_all.jsonl \
  --model gpt-5 \
  --max-follow-ups 2 \
  --max-concurrent 100 \
  --n 3
```

### 2. Judge Responses

HalluHard supports two judging modes:

A) Claim-based verification (`--type webscraper`)

Extracts atomic claims (citation + supported content) per turn, searches the web, and judges claims against retrieved evidence. This is intended for tasks that require citation grounding.

```bash
# Evaluate using web scraper method
pixi run python -m judging_pipeline.run_pipeline \
  --input "research_questions/results/conversations_gpt-5_250convs.jsonl" \
  --type webscraper \
  --seed 42 \
  --base_path "research_questions" \
  --task research_questions \
  --max_claims_per_turn 5 \
  --n 100
```

B) Response-based verification (`--type coding_direct`)

Directly evaluates coding-task responses using a coding-specific judge (e.g., checking package installation/importing and function calling behaviors). This mode is intended for the coding task.

```bash
# Evaluate coding task using direct coding judge
pixi run python -m judging_pipeline.run_pipeline \
  --input "coding/results/conversations_gpt-5_200convs.jsonl" \
  --type coding_direct \
  --task coding \
```

### 3. Generate Reports

Generate an HTML report from an evaluation output file:

```bash
pixi run report \
  --task research_questions \
  --input "research_questions/results/conversations_gpt-5_250convs_eval_webscraper.jsonl"
```

## Available Tasks

- `research_questions` - Academic research question claims
- `legal_cases` - Legal case citations and facts
- `medical_guidelines` - Medical guideline claims
- `coding` - Code implementation claims

Each task follows the same workflow: 

**data preparation → response generation → judging → reporting**

## Evaluated Models

The framework supports multiple LLM providers and models:

- **OpenAI**: `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-medium`, `gpt-5.2`, `gpt-5.2-medium-websearch`
- **Anthropic**: `claude-opus-4-5`, `claude-sonnet-4-5`, `claude-haiku-4-5`, `claude-opus-4-5-websearch`
- **DeepSeek**: `deepseek-reasoner`, `deepseek-chat`
- **Google**: `gemini-3-pro`, `gemini-3-flash`
- **Moonshot**: `kimi-k2-thinking`
- **Z.ai**: `GLM-4.7-thinking`

## Project Structure

```
<task>/
  ├── data/                    # Input data
  │   └── *.jsonl             # Task-specific question datasets
  ├── results/                 # Generated conversations and evaluations
  │   ├── conversations_<model>_<n>convs.jsonl
  │   ├── conversations_<model>_<n>convs_eval_<type>.jsonl
  │   └── reports/             # HTML reports
  ├── prompts/                 # Task-specific prompts
  └── generate_responses.py    # Response generation script
```

## CLI Reference

### Response generation parameters
- `--data`: Path to input data file
- `--model`: Model name to use
- `--max-follow-ups`: Number of follow-up questions per conversation (typically 2)
- `--max-concurrent`: Number of concurrent API requests (varies by model rate limits)
- `--n`: Number of conversations to generate (optional, defaults to all)
- `--output`: Custom output path (optional)

### Judging pipeline parameters
- `--input`: Path to conversations file to evaluate
- `--type`: Evaluation method (`webscraper` or `coding_direct`)
- `--seed`: Random seed for reproducibility
- `--base_path`: Base directory for task
- `--task`: Task name
- `--max_claims_per_turn`: Maximum claims per turn (typically 5)
- `--n`: Number of conversations to evaluate (optional)
- Worker parameters: `--searchers`, `--fetchers`, `--filters`, `--judges`

### Report generation parameters
- `--task`: Task name
- `--input`: Path to evaluation results file

## Launch Experiments

See [final_run.sh](final_run.sh) for the final launches used in the paper.

## (Optional) Data Generation

For full transparency, we provide our data generation pipelines under `<task_name>/data_fetcher.py`. Readers are welcome to re-use this script to generate more questions for their own use. 

## Citing this work
If you find our code useful, please cite our work
```
@misc{fan2026halluhardhardmultiturnhallucination,
      title={HalluHard: A Hard Multi-Turn Hallucination Benchmark}, 
      author={Dongyang Fan and Sebastien Delsad and Nicolas Flammarion and Maksym Andriushchenko},
      year={2026},
      eprint={2602.01031},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.01031}, 
}
```

