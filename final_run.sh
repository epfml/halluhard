RESEARCH_DATA_PATH="research_questions/data/research_questions_all.jsonl"
LAW_DATA_PATH="legal_cases/data/legal_cases_all.jsonl"
CODING_DATA_PATH="coding/data/coding_questions.jsonl"
MEDICAL_DATA_PATH="medical_guidelines/data/guidelines.jsonl"    


###### LEGAL CASES

task_name="legal_cases"

# responses generation
# pixi run python -m $task_name.generate_responses --data $LAW_DATA_PATH --model gemini-3-pro --max-follow-ups 2 --max-concurrent 10
# pixi run python -m $task_name.generate_responses --data $LAW_DATA_PATH --model gemini-3-flash --max-follow-ups 2 --max-concurrent 10
# pixi run python -m $task_name.generate_responses --data $LAW_DATA_PATH --model kimi-k2-thinking --max-follow-ups 2 --max-concurrent 10 
# pixi run python -m $task_name.generate_responses --data $LAW_DATA_PATH --model gpt-5-nano --max-follow-ups 2 --max-concurrent 100
# pixi run python -m $task_name.generate_responses --data $LAW_DATA_PATH --model kimi-k2-thinking --max-follow-ups 2 --max-concurrent 100 
# pixi run python -m $task_name.generate_responses --data $LAW_DATA_PATH --model gpt-5 --max-follow-ups 2 --max-concurrent 100
# pixi run python -m $task_name.generate_responses --data $LAW_DATA_PATH --model gpt-5-mini --max-follow-ups 2 --max-concurrent 100
# pixi run python -m $task_name.generate_responses --data $LAW_DATA_PATH --model gpt-5.2 --max-follow-ups 2 --max-concurrent 100
# pixi run python -m $task_name.generate_responses --data $LAW_DATA_PATH --model claude-sonnet-4-5 --max-follow-ups 2 --max-concurrent 100 
# pixi run python -m $task_name.generate_responses --data $LAW_DATA_PATH --model deepseek-reasoner --max-follow-ups 2 --max-concurrent 100
# pixi run python -m $task_name.generate_responses --data $LAW_DATA_PATH --model claude-opus-4-5 --max-follow-ups 2 --max-concurrent 30 
# pixi run python -m $task_name.generate_responses --data $LAW_DATA_PATH --model deepseek-chat --max-follow-ups 2 --max-concurrent 100
# pixi run python -m $task_name.generate_responses --data $LAW_DATA_PATH --model claude-haiku-4-5 --max-follow-ups 2 --max-concurrent 100 
# pixi run python -m $task_name.generate_responses --data $LAW_DATA_PATH --model gpt-5-medium --max-follow-ups 2 --max-concurrent 100
# pixi run python -m $task_name.generate_responses --data $LAW_DATA_PATH --model gpt-5.2-medium-websearch --max-follow-ups 2 --max-concurrent 100
# pixi run python -m $task_name.generate_responses --data $LAW_DATA_PATH --model claude-opus-4-5-websearch --max-follow-ups 2 --max-concurrent 30
# pixi run python -m $task_name.generate_responses --data $LAW_DATA_PATH --model glm-4.7-thinking --max-follow-ups 2 --max-concurrent 50



# judge the responses
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gemini-3-flash_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gemini-3-pro_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_claude-opus-4-5-websearch_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gpt-5.2-medium-websearch_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gpt-5-nano_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gpt-5-mini_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gpt-5_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gpt-5-medium_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gpt-5.2_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_claude-sonnet-4-5_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_claude-opus-4-5_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_claude-haiku-4-5_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_deepseek-reasoner_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_deepseek-chat_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_kimi-k2-thinking_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_glm-4.7-thinking_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100

# report 
# pixi run report --task $task_name --input "$task_name/results/conversations_gemini-3-flash_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gemini-3-pro_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_claude-opus-4-5-websearch_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gpt-5.2-medium-websearch_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gpt-5-nano_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gpt-5-mini_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gpt-5_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_deepseek-reasoner_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_claude-sonnet-4-5_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gpt-5-medium_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gpt-5.2_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_claude-opus-4-5_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_claude-haiku-4-5_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_deepseek-chat_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_kimi-k2-thinking_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_glm-4.7-thinking_250convs_eval_webscraper.jsonl"

###### RESEARCH QUESTIONS

task_name="research_questions"

# responses generation
# pixi run python -m $task_name.generate_responses --data $RESEARCH_DATA_PATH --model gemini-3-flash --max-follow-ups 2 --max-concurrent 10 
# pixi run python -m $task_name.generate_responses --data $RESEARCH_DATA_PATH --model claude-opus-4-5-websearch --max-follow-ups 2 --max-concurrent 10 
# pixi run python -m $task_name.generate_responses --data $RESEARCH_DATA_PATH --model gemini-3-pro --max-follow-ups 2 --max-concurrent 10
# pixi run python -m $task_name.generate_responses --data $RESEARCH_DATA_PATH --model gpt-5-nano --max-follow-ups 2 --max-concurrent 100
# pixi run python -m $task_name.generate_responses --data $RESEARCH_DATA_PATH --model gpt-5 --max-follow-ups 2 --max-concurrent 100
# pixi run python -m $task_name.generate_responses --data $RESEARCH_DATA_PATH --model gpt-5-mini --max-follow-ups 2 --max-concurrent 100
# pixi run python -m $task_name.generate_responses --data $RESEARCH_DATA_PATH --model gpt-5.2 --max-follow-ups 2 --max-concurrent 100
# pixi run python -m $task_name.generate_responses --data $RESEARCH_DATA_PATH --model claude-sonnet-4-5 --max-follow-ups 2 --max-concurrent 100 
# pixi run python -m $task_name.generate_responses --data $RESEARCH_DATA_PATH --model deepseek-reasoner --max-follow-ups 2 --max-concurrent 100
# pixi run python -m $task_name.generate_responses --data $RESEARCH_DATA_PATH --model claude-opus-4-5 --max-follow-ups 2 --max-concurrent 30 
# pixi run python -m $task_name.generate_responses --data $RESEARCH_DATA_PATH --model deepseek-chat --max-follow-ups 2 --max-concurrent 100
# pixi run python -m $task_name.generate_responses --data $RESEARCH_DATA_PATH --model claude-haiku-4-5 --max-follow-ups 2 --max-concurrent 100 
# pixi run python -m $task_name.generate_responses --data $RESEARCH_DATA_PATH --model gpt-5-medium --max-follow-ups 2 --max-concurrent 100
# pixi run python -m $task_name.generate_responses --data $RESEARCH_DATA_PATH --model gpt-5.2-medium-websearch --max-follow-ups 2 --max-concurrent 100
# pixi run python -m $task_name.generate_responses --data $RESEARCH_DATA_PATH --model kimi-k2-thinking --max-follow-ups 2 --max-concurrent 100
# pixi run python -m $task_name.generate_responses --data $RESEARCH_DATA_PATH --model glm-4.7-thinking --max-follow-ups 2 --max-concurrent 50


# judge the responses
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gemini-3-flash_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gemini-3-pro_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_claude-opus-4-5-websearch_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gpt-5.2-medium-websearch_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gpt-5-nano_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gpt-5-mini_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gpt-5_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gpt-5-medium_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gpt-5.2_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_claude-sonnet-4-5_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_claude-opus-4-5_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_claude-haiku-4-5_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_deepseek-reasoner_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_deepseek-chat_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_kimi-k2-thinking_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_glm-4.7-thinking_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100

# report 
# pixi run report --task $task_name --input "$task_name/results/conversations_gemini-3-flash_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gemini-3-pro_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_claude-opus-4-5-websearch_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gpt-5.2-medium-websearch_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gpt-5-nano_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gpt-5-mini_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gpt-5_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gpt-5-medium_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gpt-5.2_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_claude-sonnet-4-5_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_claude-opus-4-5_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_claude-haiku-4-5_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_deepseek-reasoner_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_deepseek-chat_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_kimi-k2-thinking_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_glm-4.7-thinking_250convs_eval_webscraper.jsonl"


###### MEDICAL GUIDELINES

task_name="medical_guidelines"

#responses generation

# responses generation
# pixi run python -m medical_guidelines.generate_responses --data $MEDICAL_DATA_PATH --model gpt-5-nano --max-follow-ups 2 --max-concurrent 100
# pixi run python -m medical_guidelines.generate_responses --data $MEDICAL_DATA_PATH --model gpt-5-mini --max-follow-ups 2 --max-concurrent 100
# pixi run python -m medical_guidelines.generate_responses --data $MEDICAL_DATA_PATH --model gpt-5 --max-follow-ups 2 --max-concurrent 100
# pixi run python -m medical_guidelines.generate_responses --data $MEDICAL_DATA_PATH --model gpt-5-medium --max-follow-ups 2 --max-concurrent 100 
# pixi run python -m medical_guidelines.generate_responses --data $MEDICAL_DATA_PATH --model gpt-5.2 --max-follow-ups 2 --max-concurrent 100 
# pixi run python -m medical_guidelines.generate_responses --data $MEDICAL_DATA_PATH --model gpt-5.2-medium-websearch --max-follow-ups 2 --max-concurrent 100
# pixi run python -m medical_guidelines.generate_responses --data $MEDICAL_DATA_PATH --model claude-sonnet-4-5 --max-follow-ups 2 --max-concurrent 100 
# pixi run python -m medical_guidelines.generate_responses --data $MEDICAL_DATA_PATH --model claude-opus-4-5 --max-follow-ups 2 --max-concurrent 30 
# pixi run python -m medical_guidelines.generate_responses --data $MEDICAL_DATA_PATH --model claude-opus-4-5-websearch --max-follow-ups 2 --max-concurrent 30
# pixi run python -m medical_guidelines.generate_responses --data $MEDICAL_DATA_PATH --model claude-haiku-4-5 --max-follow-ups 2 --max-concurrent 100 
# pixi run python -m medical_guidelines.generate_responses --data $MEDICAL_DATA_PATH --model deepseek-reasoner --max-follow-ups 2 --max-concurrent 100
# pixi run python -m medical_guidelines.generate_responses --data $MEDICAL_DATA_PATH --model deepseek-chat --max-follow-ups 2 --max-concurrent 100
# pixi run python -m medical_guidelines.generate_responses --data $MEDICAL_DATA_PATH --model gemini-3-pro --max-follow-ups 2 --max-concurrent 10
# pixi run python -m medical_guidelines.generate_responses --data $MEDICAL_DATA_PATH --model gemini-3-flash --max-follow-ups 2 --max-concurrent 10
# pixi run python -m medical_guidelines.generate_responses --data $MEDICAL_DATA_PATH --model kimi-k2-thinking --max-follow-ups 2 --max-concurrent 100
# pixi run python -m medical_guidelines.generate_responses --data $MEDICAL_DATA_PATH --model glm-4.7-thinking --max-follow-ups 2 --max-concurrent 50  --n 150


# judge the responses
# pixi run python -m judging_pipeline.run_pipeline --input "medical_guidelines/results/conversations_gemini-3-flash_250convs.jsonl" --type webscraper --seed 42 --base_path "medical_guidelines" --task medical_guidelines --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "medical_guidelines/results/conversations_gemini-3-pro_250convs.jsonl" --type webscraper --seed 42 --base_path "medical_guidelines" --task medical_guidelines --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "medical_guidelines/results/conversations_claude-opus-4-5-websearch_250convs.jsonl" --type webscraper --seed 42 --base_path "medical_guidelines" --task medical_guidelines --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "medical_guidelines/results/conversations_gpt-5.2-medium-websearch_250convs.jsonl" --type webscraper --seed 42 --base_path "medical_guidelines" --task medical_guidelines --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "medical_guidelines/results/conversations_gpt-5-nano_250convs.jsonl" --type webscraper --seed 42 --base_path "medical_guidelines" --task medical_guidelines --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "medical_guidelines/results/conversations_gpt-5-mini_250convs.jsonl" --type webscraper --seed 42 --base_path "medical_guidelines" --task medical_guidelines --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "medical_guidelines/results/conversations_gpt-5_250convs.jsonl" --type webscraper --seed 42 --base_path "medical_guidelines" --task medical_guidelines --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "medical_guidelines/results/conversations_gpt-5-medium_250convs.jsonl" --type webscraper --seed 42 --base_path "medical_guidelines" --task medical_guidelines --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "medical_guidelines/results/conversations_gpt-5.2_250convs.jsonl" --type webscraper --seed 42 --base_path "medical_guidelines" --task medical_guidelines --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "medical_guidelines/results/conversations_claude-sonnet-4-5_250convs.jsonl" --type webscraper --seed 42 --base_path "medical_guidelines" --task medical_guidelines --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "medical_guidelines/results/conversations_claude-opus-4-5_250convs.jsonl" --type webscraper --seed 42 --base_path "medical_guidelines" --task medical_guidelines --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "medical_guidelines/results/conversations_claude-haiku-4-5_250convs.jsonl" --type webscraper --seed 42 --base_path "medical_guidelines" --task medical_guidelines --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "medical_guidelines/results/conversations_deepseek-reasoner_250convs.jsonl" --type webscraper --seed 42 --base_path "medical_guidelines" --task medical_guidelines --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "medical_guidelines/results/conversations_deepseek-chat_250convs.jsonl" --type webscraper --seed 42 --base_path "medical_guidelines" --task medical_guidelines --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "medical_guidelines/results/conversations_kimi-k2-thinking_250convs.jsonl" --type webscraper --seed 42 --base_path "medical_guidelines" --task medical_guidelines --max_claims_per_turn 5 --n 100
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_glm-4.7-thinking_250convs.jsonl" --type webscraper --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5 --n 100

# report 
# pixi run report --task $task_name --input "medical_guidelines/results/conversations_gemini-3-flash_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "medical_guidelines/results/conversations_gemini-3-pro_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "medical_guidelines/results/conversations_claude-opus-4-5-websearch_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "medical_guidelines/results/conversations_gpt-5.2-medium-websearch_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "medical_guidelines/results/conversations_gpt-5-nano_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "medical_guidelines/results/conversations_gpt-5-mini_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "medical_guidelines/results/conversations_gpt-5_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "medical_guidelines/results/conversations_gpt-5-medium_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "medical_guidelines/results/conversations_gpt-5.2_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "medical_guidelines/results/conversations_claude-sonnet-4-5_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "medical_guidelines/results/conversations_claude-opus-4-5_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "medical_guidelines/results/conversations_claude-haiku-4-5_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "medical_guidelines/results/conversations_deepseek-reasoner_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "medical_guidelines/results/conversations_deepseek-chat_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "medical_guidelines/results/conversations_kimi-k2-thinking_250convs_eval_webscraper.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_glm-4.7-thinking_250convs_eval_webscraper.jsonl"


###### CODING

task_name="coding"

# responses generation
# pixi run python -m $task_name.generate_responses --data $CODING_DATA_PATH --model gpt-5-nano --max-follow-ups 2 --max-concurrent 100 --n 50
# pixi run python -m $task_name.generate_responses --data $CODING_DATA_PATH --model gpt-5 --max-follow-ups 2 --max-concurrent 100 --n 50
# pixi run python -m $task_name.generate_responses --data $CODING_DATA_PATH --model gpt-5-mini --max-follow-ups 2 --max-concurrent 100 --n 50
# pixi run python -m $task_name.generate_responses --data $CODING_DATA_PATH --model gpt-5.2 --max-follow-ups 2 --max-concurrent 100 --n 50
# pixi run python -m $task_name.generate_responses --data $CODING_DATA_PATH --model claude-sonnet-4-5 --max-follow-ups 2 --max-concurrent 100 --n 50
# pixi run python -m $task_name.generate_responses --data $CODING_DATA_PATH --model deepseek-reasoner --max-follow-ups 2 --max-concurrent 100 --n 50
# pixi run python -m $task_name.generate_responses --data $CODING_DATA_PATH --model claude-opus-4-5 --max-follow-ups 2 --max-concurrent 30 --n 50 
# pixi run python -m $task_name.generate_responses --data $CODING_DATA_PATH --model deepseek-chat --max-follow-ups 2 --max-concurrent 100 --n 50
# pixi run python -m $task_name.generate_responses --data $CODING_DATA_PATH --model claude-haiku-4-5 --max-follow-ups 2 --max-concurrent 100 --n 50
# pixi run python -m $task_name.generate_responses --data $CODING_DATA_PATH --model gpt-5-medium --max-follow-ups 2 --max-concurrent 100 --n 50
# pixi run python -m $task_name.generate_responses --data $CODING_DATA_PATH --model kimi-k2-thinking --max-follow-ups 2 --max-concurrent 100 --n 50
# pixi run python -m $task_name.generate_responses --data $CODING_DATA_PATH --model gemini-3-flash --max-follow-ups 2 --max-concurrent 100 --n 50
# pixi run python -m $task_name.generate_responses --data $CODING_DATA_PATH --model gemini-3-pro --max-follow-ups 2 --max-concurrent 100 --n 50
# pixi run python -m $task_name.generate_responses --data $CODING_DATA_PATH --model claude-opus-4-5-websearch --max-follow-ups 2 --max-concurrent 100 --n 50
# pixi run python -m $task_name.generate_responses --data $CODING_DATA_PATH --model gpt-5.2-medium-websearch --max-follow-ups 2 --max-concurrent 100 --n 50
# pixi run python -m $task_name.generate_responses --data $CODING_DATA_PATH --model glm-4.7-thinking --max-follow-ups 2 --max-concurrent 50 --n 50


# judge the responses
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gemini-3-flash_200convs.jsonl" --type coding_direct --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gemini-3-pro_200convs.jsonl" --type coding_direct --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_claude-opus-4-5-websearch_200convs.jsonl" --type coding_direct --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gpt-5.2-medium-websearch_200convs.jsonl" --type coding_direct --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gpt-5-nano_200convs.jsonl" --type coding_direct --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gpt-5-mini_200convs.jsonl" --type coding_direct --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gpt-5_200convs.jsonl" --type coding_direct --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gpt-5-medium_200convs.jsonl" --type coding_direct --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_gpt-5.2_200convs.jsonl" --type coding_direct --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_claude-sonnet-4-5_200convs.jsonl" --type coding_direct --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_claude-opus-4-5_200convs.jsonl" --type coding_direct --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_claude-haiku-4-5_200convs.jsonl" --type coding_direct --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_deepseek-reasoner_200convs.jsonl" --type coding_direct --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_deepseek-chat_200convs.jsonl" --type coding_direct --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_kimi-k2-thinking_200convs.jsonl" --type coding_direct --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5
# pixi run python -m judging_pipeline.run_pipeline --input "$task_name/results/conversations_glm-4.7-thinking_200convs.jsonl" --type coding_direct --seed 42 --base_path "$task_name" --task $task_name --max_claims_per_turn 5

# # report 
# pixi run report --task $task_name --input "$task_name/results/conversations_gemini-3-flash_200convs_eval_coding_direct.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gemini-3-pro_200convs_eval_coding_direct.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_claude-opus-4-5-websearch_200convs_eval_coding_direct.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gpt-5.2-medium-websearch_200convs_eval_coding_direct.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gpt-5-nano_200convs_eval_coding_direct.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gpt-5-mini_200convs_eval_coding_direct.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gpt-5_200convs_eval_coding_direct.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gpt-5-medium_200convs_eval_coding_direct.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_gpt-5.2_200convs_eval_coding_direct.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_claude-sonnet-4-5_200convs_eval_coding_direct.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_claude-opus-4-5_200convs_eval_coding_direct.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_claude-haiku-4-5_200convs_eval_coding_direct.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_deepseek-reasoner_200convs_eval_coding_direct.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_deepseek-chat_200convs_eval_coding_direct.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_kimi-k2-thinking_200convs_eval_coding_direct.jsonl"
# pixi run report --task $task_name --input "$task_name/results/conversations_glm-4.7-thinking_200convs_eval_coding_direct.jsonl"
