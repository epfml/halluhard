# %%
import pandas as pd
import json
from pathlib import Path

# Load both annotation files
def load_claims(filepath):
    """Load claims from JSONL file."""
    claims = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                claims.append(json.loads(line))
    return claims

# File paths
human1_path = Path("../responses_human1_annotation_new_format.jsonl")
human2_path = Path("../responses_human2_annotation_new_format.jsonl")

# Load claims
human1_claims = load_claims(human1_path)
human2_claims = load_claims(human2_path)

print(f"human1 claims: {len(human1_claims)}")
print(f"human2 claims: {len(human2_claims)}")

# %%
# Group claims by (conversation_id, turn_number)
from collections import defaultdict
from openai import OpenAI
import re

client = OpenAI()

def group_claims_by_turn(claims):
    """Group claims by (conversation_id, turn_number)."""
    groups = defaultdict(list)
    for claim in claims:
        key = (claim.get("conversation_id", 0), claim.get("turn_number", 0))
        groups[key].append(claim)
    return groups

human1_groups = group_claims_by_turn(human1_claims)
human2_groups = group_claims_by_turn(human2_claims)

# Get all unique (conv_id, turn_nb) keys
all_keys = set(human1_groups.keys()) | set(human2_groups.keys())
print(f"Unique (conv_id, turn_nb) groups: {len(all_keys)}")
print(f"human1 groups: {len(human1_groups)}")
print(f"human2 groups: {len(human2_groups)}")


# %%
# Use GPT-5.1 to match claims within each (conv_id, turn_nb) group

def format_claim_for_llm(idx, claim):
    """Format a claim for the LLM prompt."""
    data = claim.get("data", {})
    content = data.get("claimed_content", "")[:300]  # Truncate long content
    title = data.get("claimed_title", "")
    authors = data.get("claimed_authors", "")
    year = data.get("claimed_year", "")
    return f"{idx}. Title: {title}\n   Authors: {authors}\n   Year: {year}\n   Content: {content}..."

def match_claims_with_llm(A_claims, B_claims, conv_id, turn_nb):
    """Use GPT-5.1 to match claims between two lists."""
    if not A_claims or not B_claims:
        return [], ""
    
    # Format claims for the prompt
    A_formatted = "\n\n".join([format_claim_for_llm(i+1, c) for i, c in enumerate(A_claims)])
    B_formatted = "\n\n".join([format_claim_for_llm(i+1, c) for i, c in enumerate(B_claims)])
    
    prompt = f"""You are matching claims between two annotators for conversation {conv_id}, turn {turn_nb}.

HUMAN 1 claims:
{A_formatted}

HUMAN 2 claims:
{B_formatted}

Match claims that refer to the SAME academic source (same paper/book). They may have slightly different wording but should be about the same reference.
They must also closely match in the content of the claim. If the content of the claim is fondamentally different, then do not match them (it happens that the authors and year are the same but the content is claiming something different).

Output ONLY the matching pairs as tuples (human1_idx, human2_idx), one per line.
If a claim has no match, don't include it.

Example output format:
(1, 2)
(2, 1)
(3, 3)

Output the matches now:"""

    response = client.responses.create(
        model="gpt-5.1",
        input=[{"role": "user", "content": prompt}],
    )
    
    # Parse the response
    result_text = response.output_text.strip()
    matches = []
    pattern = r'\((\d+),\s*(\d+)\)'
    for match in re.findall(pattern, result_text):
        A_idx = int(match[0]) - 1  # Convert to 0-indexed
        B_idx = int(match[1]) - 1
        if 0 <= A_idx < len(A_claims) and 0 <= B_idx < len(B_claims):
            matches.append((A_idx, B_idx))
    
    return matches, result_text

# Process all groups IN PARALLEL using asyncio
from tqdm.asyncio import tqdm as atqdm
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def match_claims_with_llm_async(A_claims, B_claims, conv_id, turn_nb):
    """Use GPT-5-Mini to match claims between two lists (async version)."""
    if not A_claims or not B_claims:
        return (conv_id, turn_nb), [], ""
    
    # Format claims for the prompt
    A_formatted = "\n\n".join([format_claim_for_llm(i+1, c) for i, c in enumerate(A_claims)])
    B_formatted = "\n\n".join([format_claim_for_llm(i+1, c) for i, c in enumerate(B_claims)])
    
    prompt = f"""You are matching claims between two annotators for conversation {conv_id}, turn {turn_nb}.

HUMAN 1 claims:
{A_formatted}

HUMAN 2 claims:
{B_formatted}

Match claims that refer to the SAME academic source (same paper/book). They may have slightly different wording but should be about the same reference.
Match claims MUST also closely match in the content of the claim. If the content of the claim is different (it claims something different), then do not match them (it happens that the authors and year are the same but the content is claiming something different).

Output ONLY the matching pairs as tuples (human1_idx, human2_idx), one per line.
If a claim has no match (either reference or content don't closely match), do NOT include it.

Example of contents that do NOT match:
In the following example, even though the authors and year are the same, the content is claiming something different, hence you must not match them.
Claim 1:
 “Answer-making vs sense-making frames (with the listed discourse markers) as coded indicators of epistemological framing.”
*   Tuminaro, J., & Redish, E. F. (2007). Elements of a cognitive model of physics problem solving: Epistemic games. *Physical Review Special Topics - Physics Education Research, 3*(2), 020101.
Claim 2:
 Work includes explicit justification, connects ideas to principles, uses multiple representations, and shows evidence of checking for consistency (Tuminaro, 2007)
END Example.


Example output format:
(1, 2)
(2, 1)
(3, 3)

Output the matches now:"""

    response = await async_client.responses.create(
        model="gpt-5-mini",
        input=[{"role": "user", "content": prompt}],
    )
    
    # Parse the response
    result_text = response.output_text.strip()
    matches = []
    pattern = r'\((\d+),\s*(\d+)\)'
    for match in re.findall(pattern, result_text):
        A_idx = int(match[0]) - 1
        B_idx = int(match[1]) - 1
        if 0 <= A_idx < len(A_claims) and 0 <= B_idx < len(B_claims):
            matches.append((A_idx, B_idx))
    
    return (conv_id, turn_nb), matches, result_text

async def process_all_groups():
    """Process all groups in parallel."""
    tasks = []
    for key in sorted(all_keys):
        conv_id, turn_nb = key
        A_claims_group = human1_groups.get(key, [])
        B_claims_group = human2_groups.get(key, [])
        
        if A_claims_group and B_claims_group:
            tasks.append(match_claims_with_llm_async(A_claims_group, B_claims_group, conv_id, turn_nb))
    
    print(f"Matching claims using GPT-5-Mini ({len(tasks)} groups in parallel)...")
    results = await atqdm.gather(*tasks)
    
    # Build results dicts
    all_matches = {}
    llm_responses = {}
    for (conv_id, turn_nb), matches, raw_response in results:
        key = (conv_id, turn_nb)
        all_matches[key] = matches
        llm_responses[key] = raw_response
    
    # Add empty matches for groups with only one side
    for key in all_keys:
        if key not in all_matches:
            all_matches[key] = []
    
    return all_matches, llm_responses

# Run the async function
all_matches, llm_responses = await process_all_groups()

print(f"\nProcessed {len(all_keys)} groups")
total_matches = sum(len(m) for m in all_matches.values())
print(f"Total matches found: {total_matches}")


# %%
# Build matched_claims, human1_only, and human2_only from LLM matches

matched_claims = []  # Claims found in both
human1_only = []  # Claims in human1 but not matched
human2_only = []   # Claims in human2 but not matched

for key in sorted(all_keys):
    conv_id, turn_nb = key
    A_claims_group = human1_groups.get(key, [])
    B_claims_group = human2_groups.get(key, [])
    matches = all_matches.get(key, [])
    
    # Track which indices were matched
    matched_A_indices = set()
    matched_B_indices = set()
    
    # Process matches
    for A_idx, B_idx in matches:
        matched_A_indices.add(A_idx)
        matched_B_indices.add(B_idx)
        
        A_claim = A_claims_group[A_idx]
        B_claim = B_claims_group[B_idx]
        
        A_data = A_claim.get("data", {})
        B_data = B_claim.get("data", {})
        
        merged = {
            "claim_id": A_claim.get("claim_id", ""),
            "conversation_id": conv_id,
            "turn_number": turn_nb,
            # Claim content (from human1)
            "human1_claimed_content": A_data.get("claimed_content", ""),
            "human1_claimed_title": A_data.get("claimed_title", ""),
            "human1_claimed_authors": A_data.get("claimed_authors", ""),
            "human1_claimed_year": A_data.get("claimed_year", ""),
            "human1_inferred_source_type": A_data.get("inferred_source_type", ""),

            "human2_claimed_content": B_data.get("claimed_content", ""),    
            "human2_claimed_title": B_data.get("claimed_title", ""),
            "human2_claimed_authors": B_data.get("claimed_authors", ""),
            "human2_claimed_year": B_data.get("claimed_year", ""),
            "human2_inferred_source_type": B_data.get("inferred_source_type", ""),
            
            # Human1 (human1) annotations
            "human1_ref_grounding": A_data.get("human_reference_grounding", ""),
            "human1_content_grounding": A_data.get("human_content_grounding", ""),
            "human1_comment": A_data.get("human_comment", ""),
            # Human2 (human2) annotations
            "human2_ref_grounding": B_data.get("human_reference_grounding", ""),
            "human2_content_grounding": B_data.get("human_content_grounding", ""),
            "human2_comment": B_data.get("human_comment", ""),
        }

        # IF human1 or human2 have empty ref grounding or content grounding, then ignore the match
        if merged["human1_ref_grounding"] == "" or merged["human2_ref_grounding"] == "" or merged["human1_content_grounding"] == "" or merged["human2_content_grounding"] == "":
            continue

        matched_claims.append(merged)
    
    # Collect unmatched human1 claims
    for i, claim in enumerate(A_claims_group):
        if i not in matched_A_indices:
            human1_only.append(claim)
    
    # Collect unmatched human2 claims
    for i, claim in enumerate(B_claims_group):
        if i not in matched_B_indices:
            human2_only.append(claim)

# Create DataFrame
df_matched = pd.DataFrame(matched_claims)
df_matched = df_matched.sort_values(["conversation_id", "turn_number"]).reset_index(drop=True)

print("=" * 80)
print("MATCHED CLAIMS (Intersection via LLM)")
print("=" * 80)
print(f"Total matched: {len(df_matched)} claims")
print(f"Human 1 only (unmatched): {len(human1_only)}")
print(f"Human 2 only (unmatched): {len(human2_only)}")
print(f"\nColumns: {df_matched.columns.tolist()}")
df_matched


# %%
# Count number of disagreements
disagreements = df_matched[
    (df_matched["human1_ref_grounding"] != df_matched["human2_ref_grounding"]) | 
    (df_matched["human1_content_grounding"] != df_matched["human2_content_grounding"])
]
print(f"\nTotal disagreements: {len(disagreements)}")


# %%
id = 7

print('Conversation ID: ', disagreements.iloc[id]['conversation_id'])
print('Turn number: ', disagreements.iloc[id]['turn_number'])
print('Human 1')
print('Claimed content: \n', disagreements.iloc[id]['human1_claimed_content'])
print('Claimed title: ', disagreements.iloc[id]['human1_claimed_title'])
print('Claimed authors: ', disagreements.iloc[id]['human1_claimed_authors'])
print('Claimed year: ', disagreements.iloc[id]['human1_claimed_year'])
print('Human 1 reference grounding: ', disagreements.iloc[id]['human1_ref_grounding'])
print('Human 1 content grounding: ', disagreements.iloc[id]['human1_content_grounding'])
print('Human 2')
print('Claimed content: \n', disagreements.iloc[id]['human2_claimed_content'])
print('Claimed title: ', disagreements.iloc[id]['human2_claimed_title'])
print('Claimed authors: ', disagreements.iloc[id]['human2_claimed_authors'])
print('Claimed year: ', disagreements.iloc[id]['human2_claimed_year'])
print('Human 2 reference grounding: ', disagreements.iloc[id]['human2_ref_grounding'])
print('Human 2 content grounding: ', disagreements.iloc[id]['human2_content_grounding'])


# %% [markdown]
# # Now match with LLM annotations

# %%
# Load judge evaluation files and match to df_matched by claim_id

def load_judge_evaluations(filepath):
    """Load judge evaluations from JSONL file and extract claim evaluations."""
    claim_evals = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                if data.get("_type") == "evaluation_result":
                    # Extract all claim evaluations from this conversation
                    for claim_eval in data.get("details", {}).get("claim_evaluations", []):
                        claim_evals.append(claim_eval)
    return claim_evals

# Load all three judge evaluations
judge_eval_dir = Path(".")
openai_evals = load_judge_evaluations(judge_eval_dir / "responses_double_human_annotated_eval_openai.jsonl")
serper_evals = load_judge_evaluations(judge_eval_dir / "responses_double_human_annotated_eval_serper.jsonl")
webscraper_evals = load_judge_evaluations(judge_eval_dir / "responses_double_human_annotated_eval_webscraper.jsonl")

print(f"OpenAI judge evaluations: {len(openai_evals)}")
print(f"Serper judge evaluations: {len(serper_evals)}")
print(f"Webscraper judge evaluations: {len(webscraper_evals)}")

# Create lookup dicts by claim_id
openai_by_claim_id = {e.get("claim_id"): e for e in openai_evals}
serper_by_claim_id = {e.get("claim_id"): e for e in serper_evals}
webscraper_by_claim_id = {e.get("claim_id"): e for e in webscraper_evals}

print(f"\nUnique claim_ids - OpenAI: {len(openai_by_claim_id)}, Serper: {len(serper_by_claim_id)}, Webscraper: {len(webscraper_by_claim_id)}")


# %%
# Merge judge evaluations with df_matched by claim_id

def extract_judge_fields(eval_dict, prefix):
    """Extract relevant judge fields with a prefix."""
    if eval_dict is None:
        return {
            f"{prefix}_ref_grounding": None,
            f"{prefix}_content_grounding": None,
            f"{prefix}_hallucination": None,
            f"{prefix}_verification_error": None,
        }
    return {
        f"{prefix}_ref_grounding": eval_dict.get("reference_grounding", ""),
        f"{prefix}_content_grounding": eval_dict.get("content_grounding", ""),
        f"{prefix}_hallucination": eval_dict.get("hallucination", ""),
        f"{prefix}_verification_error": eval_dict.get("verification_error", ""),
    }

# Add judge columns to df_matched
judge_data = []
matched_count = {"openai": 0, "serper": 0, "webscraper": 0}

for idx, row in df_matched.iterrows():
    claim_id = row["claim_id"]
    
    # Get judge evaluations for this claim_id
    openai_eval = openai_by_claim_id.get(claim_id)
    serper_eval = serper_by_claim_id.get(claim_id)
    webscraper_eval = webscraper_by_claim_id.get(claim_id)
    
    if openai_eval:
        matched_count["openai"] += 1
    if serper_eval:
        matched_count["serper"] += 1
    if webscraper_eval:
        matched_count["webscraper"] += 1
    
    # Extract fields
    judge_fields = {}
    judge_fields.update(extract_judge_fields(openai_eval, "judge_openai"))
    judge_fields.update(extract_judge_fields(serper_eval, "judge_serper"))
    judge_fields.update(extract_judge_fields(webscraper_eval, "judge_webscraper"))
    
    judge_data.append(judge_fields)

# Create DataFrame with judge data and merge with df_matched
df_judges = pd.DataFrame(judge_data)
df_with_judges = pd.concat([df_matched.reset_index(drop=True), df_judges], axis=1)

print("=" * 80)
print("MATCHED CLAIMS WITH JUDGE EVALUATIONS")
print("=" * 80)
print(f"Total rows: {len(df_with_judges)}")
print(f"Claims matched with OpenAI judge: {matched_count['openai']}")
print(f"Claims matched with Serper judge: {matched_count['serper']}")
print(f"Claims matched with Webscraper judge: {matched_count['webscraper']}")
print(f"\nNew columns added: {df_judges.columns.tolist()}")
df_with_judges


# %%
df_with_judges.columns

# %%
df_with_judges.to_csv("responses_doubled_human_annotated_with_judges.csv", index=False, encoding='utf-8')

# %%
df_with_judges_no_verification_error = df_with_judges[(df_with_judges['judge_openai_verification_error'] != 'Yes') & (df_with_judges['judge_serper_verification_error'] != 'Yes') & (df_with_judges['judge_webscraper_verification_error'] != 'Yes')]
len(df_with_judges_no_verification_error)

# %%
df_with_judges_no_verification_error = df_with_judges_no_verification_error.copy()

df_with_judges_no_verification_error["OpenAI-WS Ref Grounding"] = (
    df_with_judges_no_verification_error["judge_openai_ref_grounding"]
      .astype("string")
      .str.strip()
      .str.lower()
      .str.startswith("yes")
)

df_with_judges_no_verification_error["OpenAI-WS Content Grounding"] = (
    df_with_judges_no_verification_error["judge_openai_content_grounding"]
      .astype("string")
      .str.strip()
      .str.lower()
      .str.startswith("yes")
)

df_with_judges_no_verification_error["SAFE Ref Grounding"] = (
    df_with_judges_no_verification_error["judge_serper_ref_grounding"]
      .astype("string")
      .str.strip()
      .str.lower()
      .str.startswith("yes")
)

df_with_judges_no_verification_error["SAFE Content Grounding"] = (
    df_with_judges_no_verification_error["judge_serper_content_grounding"]
    .astype("string")
    .str.strip()
    .str.lower()
    .str.startswith("yes")
)

df_with_judges_no_verification_error["Our Judge Ref Grounding"] = (
    df_with_judges_no_verification_error["judge_webscraper_ref_grounding"]
    .astype("string")
    .str.strip()
    .str.lower()
    .str.startswith("yes")
)

df_with_judges_no_verification_error["Our Judge Content Grounding"] = (
    df_with_judges_no_verification_error["judge_webscraper_content_grounding"]
    .astype("string")
    .str.strip()
    .str.lower()
    .str.startswith("yes")
)


# %%
filtered_df_with_judges = df_with_judges_no_verification_error[['conversation_id', 
'turn_number', 
'claim_id', 
'human1_claimed_authors', 
'human1_claimed_year', 
'OpenAI-WS Ref Grounding', 
'OpenAI-WS Content Grounding', 
'SAFE Ref Grounding', 
'SAFE Content Grounding', 
'Our Judge Ref Grounding', 
'Our Judge Content Grounding', 
'human1_ref_grounding', 
'human1_content_grounding', 
'human2_ref_grounding', 
'human2_content_grounding', 
'human1_comment', 
'human2_comment']]
filtered_df_with_judges.to_csv("responses_doubled_human_annotated_with_judges_filtered.csv", index=False, encoding='utf-8')

# %%



