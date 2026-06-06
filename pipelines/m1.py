import os
import re
import time
import numpy as np
import pandas as pd
import requests
import torch
from sentence_transformers import SentenceTransformer, util
from google.colab import drive, userdata


# m1 setup notes:
# switch DATA_LANGUAGE between 'fa' and 'it'
# when you change language, also check the input file, saved bank path, and output file name
# if you switch models later, update MODEL_NAME and the output file name so runs stay easy to track
MODEL_NAME = 'mistralai/mixtral-8x22b-instruct'
DATA_LANGUAGE = 'fa'
TEST_CSV = f'{DATA_LANGUAGE}_pilot_test.csv'
BANK_PKL = f'{DATA_LANGUAGE}_bge_m3_bank.pkl'
OUTPUT_CSV = f'{DATA_LANGUAGE}_M1_results.csv'
K_EXAMPLES = 4
REQUEST_TIMEOUT = 120
ROW_SLEEP_SECONDS = 1


# this one matters too: the retrieval bank must match the language you are running
# for example, fa should use the fa encoded bank, and it should use the italian one
# same idea for the test file and final csv path
SYSTEM_PROMPT_EN = 'You are a strict data formatting AI. You MUST return ONLY the requested tags. Do not include conversational filler, greetings, or explanations.'


# task 1: first decide how problematic the article is
TASK_1_INSTRUCTION_BASE_EN = """You are an expert in framing, linguistic bias, and Information Disorder.
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


# task 2: only if severity is not none, extract the problematic spans
SPANS_INSTRUCTION_BASE_EN = """You are an expert in framing, linguistic bias, and Information Disorder.
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


# task 3: explain each extracted span using the target-language pattern
RATIONALES_INSTRUCTION_BASE_EN = """You are an expert in framing, linguistic bias, and Information Disorder.
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


GLOBAL_OVERRIDE_EN = """IMPORTANT WARNING: The examples provided above are extracted directly from raw human annotations. Some of them contain human errors, typographical mistakes, and formatting inconsistencies (e.g., duplicated spans, mismatched span/rationale counts, or extra characters). These elements are human noise and MUST NOT be imitated in your output. Use these examples ONLY to learn the logical patterns of bias. In your response, you must strictly follow the standard JSON-style format requested in the instructions."""


# if the model breaks the format once, give it one clean retry
RETRY_SPANS_PROMPT_EN = """Your previous output was invalid.
You must return ONLY a single <SPANS> block in JSON list format.
Do not write any other text. Try again:

{instance}"""


RETRY_RATIONALES_PROMPT_EN = """Your previous output was invalid.
You must return ONLY a single <RATIONALES> block in JSON list format.
Each rationale inside the list must strictly use the target language conditional structure as shown in the examples.
Do not write any other text.
If there were no spans, return exactly <RATIONALES>: ["No"]. Try again:

News Excerpt: {instance}
Spans: {spans}"""


# this part is language-sensitive, so tweak it when you swap fa and it
if DATA_LANGUAGE == 'fa':
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
        raise ValueError('openrouter_api_key not found in colab secrets. please add it first.')
    return api_key


# simple openrouter call, kept close to the old version on purpose
def call_llm(prompt_text, system_prompt, max_tokens, api_key):
    payload = {
        'model': MODEL_NAME,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt_text}
        ],
        'temperature': 0.0,
        'max_tokens': max_tokens,
        'top_p': 1.0,
        'seed': 42
    }

    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'HTTP-Referer': 'https://colab.research.google.com/',
                'Content-Type': 'application/json'
            },
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        if 'choices' in data and data['choices']:
            content_obj = data['choices'][0].get('message', {}).get('content', '')
            if content_obj is not None:
                return str(content_obj).strip()

        return 'FORMAT_ERROR: Empty field return from server connection.'
    except Exception as e:
        return f'API_ERROR: {str(e)}'


# parsers stay small and greedy so brackets inside text do not break things
def parse_severity(text):
    if text.startswith('API_ERROR') or text.startswith('FORMAT_ERROR'):
        return 'API_ERROR'
    match = re.search(r'<PREDICTED_LABEL>:\s*(none|slightly|moderately|highly)', text, re.IGNORECASE)
    return match.group(1).lower() if match else 'FORMAT_ERROR'


def parse_spans(text):
    if text.startswith('API_ERROR') or text.startswith('FORMAT_ERROR'):
        return 'API_ERROR'
    match = re.search(r'<SPANS>:\s*(\[[\s\S]*\])', text)
    return match.group(1).strip() if match else 'FORMAT_ERROR'


def parse_rationales(text):
    if text.startswith('API_ERROR') or text.startswith('FORMAT_ERROR'):
        return 'API_ERROR'
    match = re.search(r'<RATIONALES>:\s*(\[[\s\S]*\])', text)
    return match.group(1).strip() if match else 'FORMAT_ERROR'


# this builds the on-the-fly few-shot block from the retrieval bank
# if you save your bank under a different column name later, update it here too
def build_dynamic_examples_string(query_emb, bank_embs, bank_df, k):
    search_results = util.semantic_search(query_emb, bank_embs, top_k=k)[0]
    examples_str = ''

    for i, result in enumerate(search_results):
        match_row = bank_df.iloc[result['corpus_id']]
        label = str(match_row['label_lower']).lower()
        spans = str(match_row['spans'])
        rats = str(match_row['rationales'])
        text = str(match_row['text'])

        examples_str += f'--- Example {i + 1} ({label.upper()}) ---\n'
        examples_str += f'{text}\n'
        examples_str += f'<PREDICTED_LABEL>: {label}\n'
        examples_str += f'<SPANS>: {spans}\n'
        examples_str += f'<RATIONALES>: {rats}\n\n'

    return examples_str


# when there is no previous output, start from the test csv
# if an output file already exists, keep going from there
def prepare_dataframe(test_csv, output_csv):
    if os.path.exists(output_csv):
        print(f'resuming from existing output file: {output_csv}')
        df = pd.read_csv(output_csv, dtype=str).fillna('')
    else:
        print('no previous output file found, starting a fresh run')
        df = pd.read_csv(test_csv, dtype=str).fillna('')

    columns_to_add = [
        'severity_raw', 'severity_parsed',
        'spans_raw', 'spans_parsed',
        'rationales_raw', 'rationales_parsed',
        'retry_used', 'model_name'
    ]

    for col in columns_to_add:
        if col not in df.columns:
            df[col] = ''

    return df


# one row goes through retrieval first, then severity, then spans, then rationales
def process_row(row, api_key, retriever_model, device, bank_df, bank_embeddings):
    article_text = str(row['text'])
    retry_flag = False

    query_embedding = retriever_model.encode(
        article_text,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=device
    )
    dynamic_examples = build_dynamic_examples_string(
        query_embedding,
        bank_embeddings,
        bank_df,
        K_EXAMPLES
    )

    prompt_1 = (
        f'{TASK_1_INSTRUCTION_BASE_EN}\n\n'
        f'{dynamic_examples}\n'
        f'{GLOBAL_OVERRIDE_EN}\n\n'
        f'Now process the following input:\n{article_text}'
    )
    sev_raw = call_llm(prompt_1, SYSTEM_PROMPT_EN, 50, api_key)
    sev_parsed = parse_severity(sev_raw)

    if sev_parsed == 'none':
        print("severity is 'none', so spans and rationales are skipped")
        spans_raw = 'SKIPPED_DUE_TO_NONE_SEVERITY'
        spans_parsed = '["No"]'
        rats_raw = 'SKIPPED_DUE_TO_NONE_SEVERITY'
        rats_parsed = '["No"]'
    else:
        prompt_2a = (
            f'{SPANS_INSTRUCTION_BASE_EN}\n\n'
            f'{dynamic_examples}\n'
            f'{GLOBAL_OVERRIDE_EN}\n\n'
            f'Now process the following input:\n{article_text}'
        )
        spans_raw = call_llm(prompt_2a, SYSTEM_PROMPT_EN, 300, api_key)
        spans_parsed = parse_spans(spans_raw)

        if spans_parsed == 'FORMAT_ERROR':
            print('span format error, trying one retry')
            retry_flag = True
            prompt_retry = RETRY_SPANS_PROMPT_EN.format(instance=article_text)
            spans_raw_retry = call_llm(prompt_retry, SYSTEM_PROMPT_EN, 300, api_key)
            spans_raw = f'ATTEMPT 1:\n{spans_raw}\n\nATTEMPT 2:\n{spans_raw_retry}'
            spans_parsed = parse_spans(spans_raw_retry)
            if spans_parsed == 'FORMAT_ERROR':
                spans_parsed = '["FORMAT_ERROR"]'
        elif spans_raw.startswith('API_ERROR'):
            spans_parsed = '["API_ERROR"]'

        if spans_parsed in ['["API_ERROR"]', '["FORMAT_ERROR"]']:
            rats_raw = 'SKIPPED_DUE_TO_SPANS_ERROR'
            rats_parsed = 'SKIPPED_DUE_TO_SPANS_ERROR'
        else:
            task_2b_compiled_prompt = RATIONALES_INSTRUCTION_BASE_EN.format(
                target_syntax_instruction=syntax_instruction,
                target_syntax_placeholder=syntax_placeholder
            )
            prompt_2b = (
                f'{task_2b_compiled_prompt}\n\n'
                f'{dynamic_examples}\n'
                f'{GLOBAL_OVERRIDE_EN}\n\n'
                f'Now process the following input:\n'
                f'News Excerpt: {article_text}\n'
                f'Spans: {spans_parsed}'
            )
            rats_raw = call_llm(prompt_2b, SYSTEM_PROMPT_EN, 800, api_key)
            rats_parsed = parse_rationales(rats_raw)

            if rats_parsed == 'FORMAT_ERROR':
                print('rationale format error, trying one retry')
                retry_flag = True
                prompt_retry_rat = RETRY_RATIONALES_PROMPT_EN.format(
                    instance=article_text,
                    spans=spans_parsed
                )
                rats_raw_retry = call_llm(prompt_retry_rat, SYSTEM_PROMPT_EN, 800, api_key)
                rats_raw = f'ATTEMPT 1:\n{rats_raw}\n\nATTEMPT 2:\n{rats_raw_retry}'
                rats_parsed = parse_rationales(rats_raw_retry)
                if rats_parsed == 'FORMAT_ERROR':
                    rats_parsed = '["FORMAT_ERROR"]'
            elif rats_raw.startswith('API_ERROR') or rats_raw.startswith('FORMAT_ERROR'):
                rats_parsed = 'FORMAT_ERROR'

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

    print(f'starting m1 retrieval run for model: {MODEL_NAME} [{DATA_LANGUAGE.upper()}]')
    print('loading bge-m3 retriever')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    retriever_model = SentenceTransformer('BAAI/bge-m3', device=device)

    print('loading exemplar bank')
    bank_df = pd.read_pickle(BANK_PKL)
    bank_embeddings = torch.from_numpy(np.array(bank_df['bge_m3_embedding'].tolist())).to(device)

    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df = prepare_dataframe(TEST_CSV, OUTPUT_CSV)

    for index, row in df.iterrows():
        if str(row.get('severity_raw', '')).strip() != '':
            continue

        print(f"\nprocessing row {index + 1} with id: {row['text_id']}")
        result = process_row(row, api_key, retriever_model, device, bank_df, bank_embeddings)

        for key, value in result.items():
            df.at[index, key] = value

        df.to_csv(OUTPUT_CSV, index=False)
        time.sleep(ROW_SLEEP_SECONDS)

    print(f'm1 run complete for [{DATA_LANGUAGE.upper()}]')


if __name__ == '__main__':
    main()