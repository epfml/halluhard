# %%
import pandas as pd

# %%
evals = pd.read_csv("responses_doubled_human_annotated_with_judges_filtered.csv", encoding="utf-8")

# %%
evals.head()

# %%
humans_ref_agree = evals[evals['human1_ref_grounding'] == evals['human2_ref_grounding']]
humans_content_agree = evals[evals['human1_content_grounding'] == evals['human2_content_grounding']]

print(f'Humans agree on ref grounding: {len(humans_ref_agree)}/{len(evals)} ({len(humans_ref_agree)/len(evals)*100:.2f}%)')
print(f'Humans agree on content grounding: {len(humans_content_agree)}/{len(evals)} ({len(humans_content_agree)/len(evals)*100:.2f}%)')

# We continue the analysis only the claims on which humans agree
# For OpenAI-WS
openaiws_agrees_with_humans = humans_ref_agree[(humans_ref_agree['OpenAI-WS Ref Grounding'] & (humans_ref_agree['human1_ref_grounding'] == 'yes')) |
    (~humans_ref_agree['OpenAI-WS Content Grounding'] & (humans_ref_agree['human1_content_grounding'] != 'yes'))]

openaiws_agrees_with_humans_content = humans_content_agree[(humans_content_agree['OpenAI-WS Content Grounding'] & (humans_content_agree['human1_content_grounding'] == 'yes')) |
    (~humans_content_agree['OpenAI-WS Content Grounding'] & (humans_content_agree['human1_content_grounding'] != 'yes'))]

print(f'OpenAI-WS agrees with humans on ref grounding: {len(openaiws_agrees_with_humans)}/{len(humans_ref_agree)} ({len(openaiws_agrees_with_humans)/len(humans_ref_agree)*100:.2f}%)')
print(f'OpenAI-WS agrees with humans on content grounding: {len(openaiws_agrees_with_humans_content)}/{len(humans_content_agree)} ({len(openaiws_agrees_with_humans_content)/len(humans_content_agree)*100:.2f}%)')

# For SAFE
safe_agrees_with_humans = humans_ref_agree[(humans_ref_agree['SAFE Ref Grounding'] & (humans_ref_agree['human1_ref_grounding'] == 'yes')) |
    (~humans_ref_agree['SAFE Content Grounding'] & (humans_ref_agree['human1_content_grounding'] != 'yes'))]

safe_agrees_with_humans_content = humans_content_agree[(humans_content_agree['SAFE Content Grounding'] & (humans_content_agree['human1_content_grounding'] == 'yes')) |
    (~humans_content_agree['SAFE Content Grounding'] & (humans_content_agree['human1_content_grounding'] != 'yes'))]

print(f'SAFE agrees with humans on ref grounding: {len(safe_agrees_with_humans)}/{len(humans_ref_agree)} ({len(safe_agrees_with_humans)/len(humans_ref_agree)*100:.2f}%)')
print(f'SAFE agrees with humans on content grounding: {len(safe_agrees_with_humans_content)}/{len(humans_content_agree)} ({len(safe_agrees_with_humans_content)/len(humans_content_agree)*100:.2f}%)')


# For Our Judge
our_judge_agrees_with_humans = humans_ref_agree[(humans_ref_agree['Our Judge Ref Grounding'] & (humans_ref_agree['human1_ref_grounding'] == 'yes')) |
    (~humans_ref_agree['Our Judge Content Grounding'] & (humans_ref_agree['human1_content_grounding'] != 'yes'))]

our_judge_agrees_with_humans_content = humans_content_agree[(humans_content_agree['Our Judge Content Grounding'] & (humans_content_agree['human1_content_grounding'] == 'yes')) |
    (~humans_content_agree['Our Judge Content Grounding'] & (humans_content_agree['human1_content_grounding'] != 'yes'))]

print(f'Our Judge agrees with humans on ref grounding: {len(our_judge_agrees_with_humans)}/{len(humans_ref_agree)} ({len(our_judge_agrees_with_humans)/len(humans_ref_agree)*100:.2f}%)')
print(f'Our Judge agrees with humans on content grounding: {len(our_judge_agrees_with_humans_content)}/{len(humans_content_agree)} ({len(our_judge_agrees_with_humans_content)/len(humans_content_agree)*100:.2f}%)')


# %%
evals_all = pd.read_csv("responses_doubled_human_annotated_with_judges.csv", encoding="utf-8")

all_humans_ref_agree = evals_all[evals_all['human1_ref_grounding'] == evals_all['human2_ref_grounding']]
all_humans_content_agree = evals_all[evals_all['human1_content_grounding'] == evals_all['human2_content_grounding']]

df_with_judges_no_verification_error = evals_all[(evals_all['judge_openai_verification_error'] != 'Yes') & (evals_all['judge_serper_verification_error'] != 'Yes') & (evals_all['judge_webscraper_verification_error'] != 'Yes')]
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

df_with_judges_no_verification_error["Human1 Ref Grounding"] = (
    df_with_judges_no_verification_error["human1_ref_grounding"]
        .astype("string").str.strip().str.lower().str.startswith("yes")
)

df_with_judges_no_verification_error["Human1 Content Grounding"] = (
    df_with_judges_no_verification_error["human1_content_grounding"]
        .astype("string").str.strip().str.lower().str.startswith("yes")
)


# %%
df = df_with_judges_no_verification_error

# Agreement with humans
our_judge_agrees_with_humans_ref = df[
    df["Our Judge Ref Grounding"] == df["Human1 Ref Grounding"]
]

our_judge_agrees_with_humans_content = df[
    df["Our Judge Content Grounding"] == df["Human1 Content Grounding"]
]

openaiws_agrees_with_humans_ref = df[
    df["OpenAI-WS Ref Grounding"] == df["Human1 Ref Grounding"]
]

openaiws_agrees_with_humans_content = df[
    df["OpenAI-WS Content Grounding"] == df["Human1 Content Grounding"]
]

# Disagreement with humans
our_judge_disagrees_with_humans_ref = df[
    df["Our Judge Ref Grounding"] != df["Human1 Ref Grounding"]
]

our_judge_disagrees_with_humans_content = df[
    df["Our Judge Content Grounding"] != df["Human1 Content Grounding"]
]

openaiws_disagrees_with_humans_ref = df[
    df["OpenAI-WS Ref Grounding"] != df["Human1 Ref Grounding"]
]

openaiws_disagrees_with_humans_content = df[
    df["OpenAI-WS Content Grounding"] != df["Human1 Content Grounding"]
]



# %%
openaiws_agrees_with_humans_judge_disagrees_ref = df[
    (df["OpenAI-WS Ref Grounding"] == df["Human1 Ref Grounding"]) &
    (df["Our Judge Ref Grounding"] != df["Human1 Ref Grounding"])
]

openaiws_agrees_with_humans_judge_disagrees_content = df[
    (df["OpenAI-WS Content Grounding"] == df["Human1 Content Grounding"]) &
    (df["Our Judge Content Grounding"] != df["Human1 Content Grounding"])
]

our_judge_agrees_with_humans_openaiws_disagrees_ref = df[
    (df["Our Judge Ref Grounding"] == df["Human1 Ref Grounding"]) &
    (df["OpenAI-WS Ref Grounding"] != df["Human1 Ref Grounding"])
]

our_judge_agrees_with_humans_openaiws_disagrees_content = df[
    (df["Our Judge Content Grounding"] == df["Human1 Content Grounding"]) &
    (df["OpenAI-WS Content Grounding"] != df["Human1 Content Grounding"])
]

safe_disagrees_with_humans_our_judge_agrees_ref = df[
    (df["SAFE Ref Grounding"] != df["Human1 Ref Grounding"]) &
    (df["Our Judge Ref Grounding"] == df["Human1 Ref Grounding"])
]

safe_disagrees_with_humans_our_judge_agrees_content = df[
    (df["SAFE Content Grounding"] != df["Human1 Content Grounding"]) &
    (df["Our Judge Content Grounding"] == df["Human1 Content Grounding"])
]

# %%
# A few examples were our judge is bad

df = openaiws_agrees_with_humans_judge_disagrees_ref.sample(4)
print('\nOpenAI-WS agrees with humans but our judge disagrees')
for i in range(len(df)):
    print(f"Example {i+1}:")
    print(f"Human1 Ref Grounding: {df.iloc[i]['Human1 Ref Grounding']}")
    print(f"Our Judge Ref Grounding: {df.iloc[i]['Our Judge Ref Grounding']}")
    print(f"OpenAI-WS Ref Grounding: {df.iloc[i]['OpenAI-WS Ref Grounding']}")

    print(f"Human1 thinking: {df.iloc[i]['human1_comment']}")
    print(f"Our Judge thinking: {df.iloc[i]['judge_webscraper_ref_grounding']}")
    print(f"OpenAI-WS thinking: {df.iloc[i]['judge_openai_ref_grounding']}")
    print("\n")

# %%
# A few examples were our judge is bad

df = openaiws_agrees_with_humans_judge_disagrees_content
print('\nOpenAI-WS agrees with humans but our judge disagrees')
for i in range(len(df)):
    print(f"Example {i+1}:")
    print(f"Human1 Content Grounding: {df.iloc[i]['Human1 Content Grounding']}")
    print(f"Our Judge Content Grounding: {df.iloc[i]['Our Judge Content Grounding']}")
    print(f"OpenAI-WS Content Grounding: {df.iloc[i]['OpenAI-WS Content Grounding']}")

    print(f"Human1 thinking: {df.iloc[i]['human1_comment']}")
    print(f"Our Judge thinking: {df.iloc[i]['judge_webscraper_content_grounding']}")
    print(f"OpenAI-WS thinking: {df.iloc[i]['judge_openai_content_grounding']}")
    print("\n")

# %%
df = our_judge_agrees_with_humans_openaiws_disagrees_content

print('\nOur judge agrees with humans but openaiws disagrees')
for i in range(len(df)):
    print(f"Example {i+1}:")
    print(f"Human1 Content Grounding: {df.iloc[i]['Human1 Content Grounding']}")
    print(f"Our Judge Content Grounding: {df.iloc[i]['Our Judge Content Grounding']}")
    print(f"OpenAI-WS Content Grounding: {df.iloc[i]['OpenAI-WS Content Grounding']}")

    print(f"Human1 thinking: {df.iloc[i]['human1_comment']}")
    print(f"Our Judge thinking: {df.iloc[i]['judge_webscraper_content_grounding']}")
    print(f"OpenAI-WS thinking: {df.iloc[i]['judge_openai_content_grounding']}")
    print("\n")

# %%
df = safe_disagrees_with_humans_our_judge_agrees_content

print('\nSAFE disagrees with our judge but agrees with humans')
for i in range(len(df)):
    print(f"Example {i+1}:")
    print(f"Human1 Content Grounding: {df.iloc[i]['Human1 Content Grounding']}")
    print(f"Our Judge Content Grounding: {df.iloc[i]['Our Judge Content Grounding']}")
    print(f"SAFE Content Grounding: {df.iloc[i]['SAFE Content Grounding']}")

    print(f"Human1 thinking: {df.iloc[i]['human1_comment']}")
    print(f"Our Judge thinking: {df.iloc[i]['judge_webscraper_content_grounding']}")
    print(f"SAFE thinking: {df.iloc[i]['judge_serper_content_grounding']}")
    print("\n")

# %%



