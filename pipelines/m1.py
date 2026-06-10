import os
import re
import time
import numpy as np
import pandas as pd
import requests
import torch
from sentence_transformers import SentenceTransformer, util
from google.colab import drive, userdata

# config:
# switch DATA_LANGUAGE between 'fa' and 'it' to change the entire setup, or even 'ru' if you want to apply to the russian subset
# when you change language, also check the model name and output file name so runs stay easy to track
DATA_LANGUAGE = "fa"  
MODEL_NAME = "mistralai/mixtral-8x22b-instruct"
TEST_CSV = f'/your_path/{DATA_LANGUAGE}_pilot_test.csv'
BANK_PKL = f'/your_path/{DATA_LANGUAGE}_bge_m3_bank.pkl'
OUTPUT_CSV = f'/your_path/{DATA_LANGUAGE}_M1_mixtral_results.csv'
K_EXAMPLES = 4
REQUEST_TIMEOUT = 120
ROW_SLEEP_SECONDS = 1

# keep the system prompt strict so the model returns only the tags we need
system_prompt_en = "You are a strict data formatting AI. You MUST return ONLY the requested tags. Do not include conversational filler, greetings, or explanations."

# task 1: severity classification template
task_1_instruction_base_en = """You are an expert in framing, linguistic bias, and Information Disorder.
Information disorder refers to the pollution of the information space, including misinformation, disinformation, and malinformation. It is often characterized by emotional manipulation, false context, or distorted framing.
Your task is to classify the severity of problematic language in a news article excerpt.

Classify the severity based on the presence and impact of the following problematic features:
- Event bias: Describing or referring to events in a way that minimizes or distorts responsibility.
- Attribution bias: Describing people, groups, or communities using emotionally charged, stereotypical, labeling, or exaggerated language.
- Sensationalism: Exaggerating or dramatizing facts.
- Speculation: Using vague or speculative statements as if they were factual.

SEVERITY LABELS:
"none" – No problematic features present. The text is factual, balanced, and neutral.
"slightly" – Minor problematic language is present, but the overall message remains mostly factual.
"moderately" – Multiple instances of bias or misleading language produce a tangible distortion that can alter the reader's understanding.
"highly" – Severe and pervasive use of biased, speculative, or emotionally manipulative language. High risk of disinformation.

OUTPUT FORMAT:
Return only one of the four labels following the tag below, exactly like this:
<PREDICTED_LABEL>: none
<PREDICTED_LABEL>: slightly
<PREDICTED_LABEL>: moderately
<PREDICTED_LABEL>: highly

Do not add explanations, comments, or any other text. Return ONLY a valid label."""

# task 2: spans extraction template
spans_instruction_base_en = """You are an expert in framing, linguistic bias, and Information Disorder.
Information disorder refers to the pollution of the information space, including misinformation, disinformation, and malinformation. It is often characterized by emotional manipulation, false context, or distorted framing.
Your task is to analyze news excerpts and identify spans of text that are misleading, biased, speculative, or emotionally charged.

Task:
- Identify ONLY unique, NON-overlapping spans.
- If no problematic spans are found, the output must be exactly <SPANS>: ["No"].

Character Preservation Rule (CRITICAL):
The extracted spans must match the original text EXACTLY, character for character.
Do not modify spelling, punctuation, capitalization, apostrophes, accents, or spaces in any way.
Special Language Rule: For Farsi text, strictly preserve all Zero-Width Non-Joiners (ZWNJ, \u200C / نیم‌فاصله). Do not replace them with spaces or delete them.

Problematic spans include:
- Event bias: Events described in a way that minimizes or distorts responsibility.
- Attribution bias: People, groups, or communities described with emotionally charged, stereotypical, labeling, or exaggerated language.
- Text that sensationalizes or dramatizes facts.
- Text that uses vague or speculative statements as if they were factual.

OUTPUT FORMAT (Strict):
If one span: <SPANS>: ["..."]
If multiple spans: <SPANS>: ["...", "..."]
If no spans: <SPANS>: ["No"]"""

# task 3: rationales generation template (contains target-language syntax placeholders)
rationales_instruction_base_en = """You are an expert in framing, linguistic bias, and Information Disorder.
Information disorder refers to the pollution of the information space, including misinformation, disinformation, and malinformation. It is often characterized by emotional manipulation, false context, or distorted framing.
Your task is to explain why specific spans of text in a news excerpt are misleading, biased, or problematic.

You are provided with a news excerpt and a list of extracted spans.
Your goal is to generate exactly ONE rationale for each span.
Strictly use the target-language conditional structure demonstrated below:
{target_syntax_instruction}

Output Instructions (Mandatory):
- Return exactly ONE rationale for each span, in the exact same order.
- If <SPANS>: ["No"], then you must return exactly <RATIONALES>: ["No"]. Do not invent rationales.
- Each rationale must be enclosed in double quotes (" ").
- All rationales must be returned in a single list, following the exact target-language placeholder format:
{target_syntax_placeholder}
- Do not use nested quotes or ellipses (...) inside the rationales.
- Do not combine multiple spans into a single rationale.
- Do not skip spans.
- Do not show your thought process, drafts, or self-corrections.
- Do not write any words outside the tags. Return ONLY the final list."""

# mitigation guideline for human annotations noise
global_override_en = """IMPORTANT WARNING: The examples provided above are extracted directly from raw human annotations. Some of them contain human errors, typographical mistakes, and formatting inconsistencies (e.g., duplicated spans, mismatched span/rationale counts, or extra characters). These elements are human noise and MUST NOT be imitated in your output. Use these examples ONLY to learn the logical patterns of bias. In your response, you must strictly follow the standard JSON-style format requested in the instructions."""

# retry prompts
retry_spans_prompt_en = """Your previous output was invalid.
You must return ONLY a single <SPANS> block in JSON list format.
Do not write any other text. Try again:

{instance}"""

retry_rationales_prompt_en = """Your previous output was invalid.
You must return ONLY a single <RATIONALES> block in JSON list format.
Each rationale inside the list must strictly use the target language conditional structure as shown in the examples.
Do not write any other text.
If there were no spans, return exactly <RATIONALES>: ["No"]. Try again:

News Excerpt: {instance}
Spans: {spans}"""


# dynamic structural syntax binders for target language mapping
if DATA_LANGUAGE == "fa":
    syntax_instruction = '"اگر [اشاره به بازه در متن], آنگاه [پیامد یا نتیجه]"'
    syntax_placeholder = '<RATIONALES>: ["اگر ..., آنگاه ...", "اگر ..., آنگاه ..."]'
else:
    syntax_instruction = '"se [riferimento allo span nel testo], allora [implicazione o conseguenza]"'
    syntax_placeholder = '<RATIONALES>: ["se ..., allora ...", "se ..., allora ..."]'


def mount_drive():
    drive.mount('/content/drive')


def load_api_key():
    api_key = userdata.get('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in Colab Secrets. Please configure it.")
    return api_key


def call_llm(prompt_text, system_prompt, max_tokens, api_key):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "top_p": 1.0,
        "seed": 42
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://colab.research.google.com/",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        if "choices" in data and len(data["choices"]) > 0:
            content_obj = data["choices"][0].get("message", {}).get("content", "")
            if content_obj is not None:
                return str(content_obj).strip()
                
        return "FORMAT_ERROR: Empty response field."
    except Exception as e:
        return f"API_ERROR: {str(e)}"


def parse_severity(text):
    if text.startswith("API_ERROR") or text.startswith("FORMAT_ERROR"):
        return "API_ERROR"
    match = re.search(r'<PREDICTED_LABEL>:\s*(none|slightly|moderately|highly)', text, re.IGNORECASE)
    return match.group(1).lower() if match else "FORMAT_ERROR"


def parse_spans(text):
    if text.startswith("API_ERROR") or text.startswith("FORMAT_ERROR"):
        return "API_ERROR"
    match = re.search(r'<SPANS>:\s*(\[[\s\S]*\])', text)
    return match.group(1).strip() if match else "FORMAT_ERROR"


def parse_rationales(text):
    if text.startswith("API_ERROR") or text.startswith("FORMAT_ERROR"):
        return "API_ERROR"
    match = re.search(r'<RATIONALES>:\s*(\[[\s\S]*\])', text)
    return match.group(1).strip() if match else "FORMAT_ERROR"


# dynamically builds the few-shot block from the retrieval bank
def build_dynamic_examples_string(query_emb, bank_embs, bank_df, k):
    search_results = util.semantic_search(query_emb, bank_embs, top_k=k)[0]
    examples_str = ""
    for i, result in enumerate(search_results):
        match_row = bank_df.iloc[result['corpus_id']]
        label = str(match_row['label_lower']).lower()
        spans = str(match_row['spans'])
        rats = str(match_row['rationales'])
        text = str(match_row['text'])

        examples_str += f"--- Example {i+1} ({label.upper()}) ---\n"
        examples_str += f"{text}\n"
        examples_str += f"<PREDICTED_LABEL>: {label}\n"
        examples_str += f"<SPANS>: {spans}\n"
        examples_str += f"<RATIONALES>: {rats}\n\n"
    return examples_str


# load output if it exists to resume, else load fresh input
def prepare_dataframe(input_csv, output_csv):
    if os.path.exists(output_csv):
        print(f"resuming tracking state from existing file: {output_csv}")
        df = pd.read_csv(output_csv, dtype=str).fillna("")
    else:
        print("no previous output file found, starting a fresh run")
        df = pd.read_csv(input_csv, dtype=str).fillna("")

    columns_to_add = [
        'severity_raw', 'severity_parsed',
        'spans_raw', 'spans_parsed',
        'rationales_raw', 'rationales_parsed',
        'retry_used', 'model_name'
    ]
    for col in columns_to_add:
        if col not in df.columns:
            df[col] = ""

    return df


# process single row with short-circuit logic and retry
def process_row(row, api_key, retriever_model, device, bank_df, bank_embeddings):
    article_text = str(row['text'])
    retry_flag = False

    # step a: cognitive semantic search
    query_embedding = retriever_model.encode(article_text, convert_to_tensor=True, normalize_embeddings=True, device=device)
    m1_dynamic_examples = build_dynamic_examples_string(query_embedding, bank_embeddings, bank_df, K_EXAMPLES)

    # step b: task 1 (severity classification)
    prompt_1 = f"{task_1_instruction_base_en}\n\n{m1_dynamic_examples}\n{global_override_en}\n\nNow process the following input:\n{article_text}"
    sev_raw = call_llm(prompt_1, system_prompt_en, 50, api_key)
    sev_parsed = parse_severity(sev_raw)

    # downward short-circuit for "none" severity cases
    if sev_parsed == "none":
        print("severity evaluated as 'none'. short-circuiting downstream layers.")
        spans_raw = "SHORT_CIRCUITED_DUE_TO_NONE_SEVERITY"
        spans_parsed = '["No"]'
        rats_raw = "SHORT_CIRCUITED_DUE_TO_NONE_SEVERITY"
        rats_parsed = '["No"]'
    else:
        # step c: task 2a (bounded spans extraction)
        prompt_2a = f"{spans_instruction_base_en}\n\n{m1_dynamic_examples}\n{global_override_en}\n\nNow process the following input:\n{article_text}"
        spans_raw = call_llm(prompt_2a, system_prompt_en, 300, api_key)
        spans_parsed = parse_spans(spans_raw)

        if spans_parsed == "FORMAT_ERROR" and not spans_raw.startswith("API_ERROR"):
            print("span format invalid. triggering retry protocol...")
            retry_flag = True
            prompt_retry = retry_spans_prompt_en.format(instance=article_text)
            spans_raw_retry = call_llm(prompt_retry, system_prompt_en, 300, api_key)
            spans_raw = f"ATTEMPT 1:\n{spans_raw}\n\nATTEMPT 2:\n{spans_raw_retry}"
            spans_parsed = parse_spans(spans_raw_retry)
            if spans_parsed == "FORMAT_ERROR":
                spans_parsed = '["FORMAT_ERROR"]'
            elif spans_raw.startswith("API_ERROR"):
                spans_parsed = '["API_ERROR"]'
        elif spans_raw.startswith("API_ERROR"):
            spans_parsed = '["API_ERROR"]'

        # step d: task 2b (conditioned rationales generation)
        if spans_parsed in ['["API_ERROR"]', '["FORMAT_ERROR"]']:
            rats_raw = "SKIPPED_DUE_TO_SPANS_ERROR"
            rats_parsed = "SKIPPED_DUE_TO_SPANS_ERROR"
        else:
            # format base english task 2b string with target language syntax bindings
            task_2b_compiled_prompt = rationales_instruction_base_en.format(
                target_syntax_instruction=syntax_instruction,
                target_syntax_placeholder=syntax_placeholder
            )
            prompt_2b = f"{task_2b_compiled_prompt}\n\n{m1_dynamic_examples}\n{global_override_en}\n\nNow process the following input:\nNews Excerpt: {article_text}\nSpans: {spans_parsed}"
            rats_raw = call_llm(prompt_2b, system_prompt_en, 800, api_key)
            rats_parsed = parse_rationales(rats_raw)

            if rats_parsed == "FORMAT_ERROR" and not rats_raw.startswith("API_ERROR"):
                print("rationale format invalid. triggering retry protocol...")
                retry_flag = True
                prompt_retry_rat = retry_rationales_prompt_en.format(instance=article_text, spans=spans_parsed)
                rats_raw_retry = call_llm(prompt_retry_rat, system_prompt_en, 800, api_key)
                rats_raw = f"ATTEMPT 1:\n{rats_raw}\n\nATTEMPT 2:\n{rats_raw_retry}"
                rats_parsed = parse_rationales(rats_raw_retry)
                if rats_parsed == "FORMAT_ERROR":
                    rats_parsed = '["FORMAT_ERROR"]'
                elif rats_raw.startswith("API_ERROR"):
                    rats_parsed = '["API_ERROR"]'
            elif rats_raw.startswith("API_ERROR"):
                rats_parsed = '["API_ERROR"]'

    return {
        'severity_raw': sev_raw,
        'severity_parsed': sev_parsed,
        'spans_raw': spans_raw,
        'spans_parsed': spans_parsed,
        'rationales_raw': rats_raw,
        'rationales_parsed': rats_parsed,
        'retry_used': str(retry_flag),
        'model_name': MODEL_NAME
    }


def main():
    mount_drive()
    api_key = load_api_key()

    print(f"starting m1 retrieval run for model: {MODEL_NAME} [{DATA_LANGUAGE.upper()}]")
    print("loading bge-m3 retriever")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    retriever_model = SentenceTransformer('BAAI/bge-m3', device=device)

    print("loading exemplar bank")
    bank_df = pd.read_pickle(BANK_PKL)
    bank_embeddings = torch.from_numpy(np.array(bank_df['bge_m3_embedding'].tolist())).to(device)

    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df = prepare_dataframe(TEST_CSV, OUTPUT_CSV)

    for index, row in df.iterrows():
        # skip rows that have already been evaluated successfully
        if pd.notna(row.get('severity_raw', "")) and str(row.get('severity_raw', "")).strip() != "":
            continue

        print(f"\nprocessing row {index + 1} with id: {row['text_id']}")
        result = process_row(row, api_key, retriever_model, device, bank_df, bank_embeddings)

        for key, value in result.items():
            df.at[index, key] = value

        df.to_csv(OUTPUT_CSV, index=False)
        time.sleep(ROW_SLEEP_SECONDS)

    print(f"m1 transactional run complete for [{DATA_LANGUAGE.upper()}].")


if __name__ == '__main__':
    main()
