# Research Questions without debugging, USING EXISTING CACHE

pixi run python -m judging_pipeline.run_pipeline --input "research_questions/results/reports/Dec 22 Queuing Judge/responses.jsonl" --type webscraper --seed 42 --base_path "research_questions"

pixi run python -m judging_pipeline.run_pipeline --input "research_questions/results/reports/Dec 22 Queuing Judge/responses.jsonl" --type serper --seed 42 --base_path "research_questions"

pixi run python -m judging_pipeline.run_pipeline --input "research_questions/results/reports/Dec 22 Queuing Judge/responses.jsonl" --type openai --seed 42 --base_path "research_questions"

# Compare Judges for Research Questions (Dec 24)
pixi run python -m tools.evaluate_judge_vs_judge_performance "research_questions/results/reports/Dec 24/responses_eval_openai.jsonl" "research_questions/results/reports/Dec 24/responses_eval_serper.jsonl" --baseline-label "OpenAI" --compare-label "Serper"
pixi run python -m tools.evaluate_judge_vs_judge_performance "research_questions/results/reports/Dec 24/responses_eval_openai.jsonl" "research_questions/results/reports/Dec 24/responses_eval_webscraper.jsonl" --baseline-label "OpenAI" --compare-label "Webscraper"


## Medical Guidelines
# Generate Data for Medical Guidelines

pixi run python -m medical_guidelines.general_data_fetcher --n 20
pixi run python medical_guidelines/generate_responses.py --data medical_guidelines/data/guidelines.jsonl --max-follow-ups 2 --max-concurrent 100

# Extract and evaluate medical guidelines, USING EXISTING CACHE

pixi run python -m judging_pipeline.run_pipeline --input "medical_guidelines/results/conversations_gpt-5-mini_19convs.jsonl" --type webscraper --seed 42 --base_path "medical_guidelines" --task medical_guidelines -n 4

pixi run python -m judging_pipeline.run_pipeline --input "medical_guidelines/results/conversations_gpt-5-mini_19convs.jsonl" --type serper --seed 42 --base_path "medical_guidelines" --task medical_guidelines -n 4

pixi run python -m judging_pipeline.run_pipeline --input "medical_guidelines/results/conversations_gpt-5-mini_19convs.jsonl" --type openai --seed 42 --base_path "medical_guidelines" --task medical_guidelines -n 4

# Generate report

pixi run report --task medical_guidelines --input "medical_guidelines/results/conversations_gpt-5-mini_19convs_eval_serper.jsonl"

# Compare Judges
pixi run python -m tools.evaluate_judge_vs_judge_performance medical_guidelines/results/conversations_gpt-5-mini_19convs_eval_openai.jsonl medical_guidelines/results/conversations_gpt-5-mini_19convs_eval_serper.jsonl --baseline-label "OpenAI" --compare-label "Serper"
pixi run python -m tools.evaluate_judge_vs_judge_performance medical_guidelines/results/conversations_gpt-5-mini_19convs_eval_openai.jsonl medical_guidelines/results/conversations_gpt-5-mini_19convs_eval_webscraper.jsonl --baseline-label "OpenAI" --compare-label "Webscraper"

## Legal Cases

# Generate initial data

pixi run python -m legal_cases.data_fetcher --n 20
pixi run python -m legal_cases.generate_responses.py --data legal_cases/data/legal_cases_all.jsonl --model gpt-5-mini

# Judge 

pixi run python -m judging_pipeline.run_pipeline --input "legal_cases/results/conversations_gpt-5-mini_4convs.jsonl" --type webscraper --seed 42 --base_path "legal_cases" --task legal_cases -n 4

pixi run python -m judging_pipeline.run_pipeline --input "legal_cases/results/conversations_gpt-5-mini_4convs.jsonl" --type serper --seed 42 --base_path "legal_cases" --task legal_cases -n 4

pixi run python -m judging_pipeline.run_pipeline --input "legal_cases/results/conversations_gpt-5-mini_4convs.jsonl" --type openai --seed 42 --base_path "legal_cases" --task legal_cases -n 4

# Generate report

pixi run report --task legal_cases --input "legal_cases/results/conversations_gpt-5-mini_4convs_eval_serper.jsonl"

# Compare Serper to OpenAI
pixi run python -m tools.evaluate_judge_vs_judge_performance legal_cases/results/conversations_gpt-5-mini_4convs_eval_openai.jsonl legal_cases/results/conversations_gpt-5-mini_4convs_eval_serper.jsonl --baseline-label "OpenAI" --compare-label "Serper"
pixi run python -m tools.evaluate_judge_vs_judge_performance legal_cases/results/conversations_gpt-5-mini_4convs_eval_openai.jsonl legal_cases/results/conversations_gpt-5-mini_4convs_eval_webscraper.jsonl --baseline-label "OpenAI" --compare-label "Webscraper"


## Coding

# Generate data

pixi run python -m coding.data_fetcher --languages Python Scala Elixir R --samples-per-language 20  --seed 42
pixi run python -m coding.generate_responses --data coding/data/coding_prompts.jsonl --model gpt-5-mini --output coding/results/conversations.jsonl --max-follow-ups 2 --max-concurrent 100

# Judge

pixi run python -m judging_pipeline.run_pipeline --input "coding/results/conversations_gpt-5-mini_80convs.jsonl" --type webscraper --seed 42 --base_path "coding" --task coding -n 4

pixi run python -m judging_pipeline.run_pipeline --input "coding/results/conversations_gpt-5-mini_80convs.jsonl" --type serper --seed 42 --base_path "coding" --task coding -n 4

pixi run python -m judging_pipeline.run_pipeline --input "coding/results/conversations_gpt-5-mini_80convs.jsonl" --type openai --seed 42 --base_path "coding" --task coding -n 4

# Generate report

pixi run report --task coding --input "coding/results/conversations_gpt-5-mini_80convs_eval_serper.jsonl"

# Compare Judges for Coding
pixi run python -m tools.evaluate_judge_vs_judge_performance coding/results/conversations_gpt-5-mini_80convs_eval_openai.jsonl coding/results/conversations_gpt-5-mini_80convs_eval_serper.jsonl --baseline-label "OpenAI" --compare-label "Serper"
pixi run python -m tools.evaluate_judge_vs_judge_performance coding/results/conversations_gpt-5-mini_80convs_eval_openai.jsonl coding/results/conversations_gpt-5-mini_80convs_eval_webscraper.jsonl --baseline-label "OpenAI" --compare-label "Webscraper"