import os
import re
import time
import ast
import json
import numpy as np
import pandas as pd
import requests
import torch
from sentence_transformers import SentenceTransformer, util
from google.colab import drive, userdata


# config:
# this file is the a1 italian setup with retrieval plus target-language prompts
# if you switch language later, also update the input file, bank path, and output file
# if you switch model later, change the output name too so runs stay easy to track
DATA_LANGUAGE = "it"
MODEL_NAME = "meta-llama/llama-4-maverick"
TEST_CSV = f'{DATA_LANGUAGE}_pilot_test.csv'
BANK_PKL = f'/{DATA_LANGUAGE}_bge_m3_bank.pkl'
OUTPUT_CSV = f'{DATA_LANGUAGE}_A1_maverick_results.csv'
K_EXAMPLES = 4
REQUEST_TIMEOUT = 120
ROW_SLEEP_SECONDS = 1


# keep the italian system prompt exactly as in the original setup
system_prompt_it = "Sei un'IA rigorosa per la formattazione dei dati. DEVI restituire SOLO i tag richiesti. Non includere riempitivi conversazionali, saluti o spiegazioni."


# task 1: severity
task_1_prompt_it = """Sei un esperto di framing, bias linguistici e Disordine dell'Informazione.
Il disordine dell'informazione è l'inquinamento dello spazio informativo che include misinformazione, disinformazione e malinformazione. È spesso caratterizzato da manipolazione emotiva, falsi contesti o framing distorto.
Il tuo compito è classificare la gravità del linguaggio problematico in un estratto di un articolo di notizie.


Classifica la gravità in base alla presenza e all’impatto delle seguenti caratteristiche problematiche:
- Bias eventivo: descrive o fa riferimento a eventi in un modo che minimizza o distorce la responsabilità.
- Bias attributivo: descrive persone, gruppi o comunità usando un linguaggio emotivamente carico, stereotipato, etichettante o esagerato.
- Sensazionalismo: esagera o drammatizza i fatti.
- Speculazione: usa affermazioni vaghe o speculative come se fossero fattuali.


ETICHETTE DI GRAVITÀ:
"none" – Nessuna caratteristica problematica presente. Il testo è fattuale, equilibrato e neutrale.
"slightly" – È presente un linguaggio leggermente problematico, ma il messaggio complessivo rimane per lo più fattuale.
"moderately" – Sono presenti più casi di bias o linguaggio fuorviante che producono una distorsione tangibile e possono alterare la comprensione del lettore.
"highly" – Uso grave e pervasivo di linguaggio parziale, speculativo o emotivamente manipolativo. Alto rischio di disinformazione.


FORMATO DI OUTPUT:
Restituisci solo una delle quattro etichette dopo il tag seguente, esattamente in questo modo:
<PREDICTED_LABEL>: none
<PREDICTED_LABEL>: slightly
<PREDICTED_LABEL>: moderately
<PREDICTED_LABEL>: highly


Non aggiungere spiegazioni, commenti o altro testo. Restituisci solo un'etichetta valida."""


# task 2: spans
spans_task_prompt_it = """Sei un esperto di framing, bias linguistici e Disordine dell'Informazione.
Il disordine dell'informazione è l'inquinamento dello spazio informativo che include misinformazione, disinformazione e malinformazione.
È spesso caratterizzato da manipolazione emotiva, falsi contesti o framing distorto.
Il tuo compito è analizzare estratti di notizie e identificare span di testo che sono fuorvianti, di parte, speculativi o emotivamente caricati.


Compito:
- Identifica solo span unici e NON sovrapposti.
- Se non trovi alcuno span problematico, l'output deve essere esattamente <SPANS>: ["No"].


Regola di conservazione dei caratteri (molto importante):
Gli span estratti devono corrispondere esattamente, carattere per carattere, al testo originale.
Non modificare in alcun modo ortografia, punteggiatura, maiuscole, apostrofi, accenti o spazi.


Gli span problematici includono:
- Bias eventivo: gli eventi sono descritti in modo da minimizzare o distorcere la responsabilità.
- Bias attributivo: persone, gruppi o comunità sono descritti con linguaggio emotivamente carico, stereotipato, etichettante o esagerato.
- Il testo sensazionalizza o drammatizza i fatti.
- Il testo usa affermazioni vaghe o speculative come se fossero fattuali.


FORMATO DI OUTPUT (rigido):
Se c'è uno span: <SPANS>: ["..."]
Se ci sono più span: <SPANS>: ["...", "..."]
Se non ci sono span: <SPANS>: ["No"]"""


# task 3: rationales
rationales_task_prompt_it = """Sei un esperto di framing, bias linguistici e Disordine dell'Informazione.
Il disordine dell'informazione è l'inquinamento dello spazio informativo che include misinformazione, disinformazione e malinformazione.
È spesso caratterizzato da manipolazione emotiva, falsi contesti o framing distorto.
Il tuo compito è spiegare perché determinati span di testo in un estratto di notizie sono fuorvianti, di parte o problematici.


Ti vengono forniti un estratto di notizia e un elenco di span estratti.
Il tuo obiettivo è generare esattamente UNA razionale per ogni span.
Usa rigorosamente questo formato:
"se [riferimento allo span nel testo], allora [implicazione o conseguenza]"


Istruzioni di output (obbligatorie):
- Restituisci esattamente una razionale per ogni span, nello stesso ordine.
- Se <SPANS>: ["No"], allora devi restituire esattamente <RATIONALES>: ["No"]. Non inventare razionali.
- Ogni razionale deve essere racchiusa tra virgolette doppie (" ").
- Tutte le razionali devono essere restituite in un'unica lista:
<RATIONALES>: ["se ..., allora ...", "se ..., allora ..."]
- Non usare virgolette annidate o puntini di sospensione (...) all'interno delle razionali.
- Non combinare più span in una sola razionale.
- Non saltare span.
- Non mostrare processo di pensiero, bozze o auto-correzioni.
- Non scrivere alcuna parola fuori dai tag.
Restituisci solo e soltanto la lista finale."""


# keep the human-noise warning exactly as it was
global_override_it = """ATTENZIONE: Gli esempi forniti sopra sono estratti direttamente da annotazioni umane e potrebbero contenere errori di formattazione (es. span duplicati, conteggi sbilanciati tra span e razionali, presenza di caratteri spuri o interruzioni di riga). Questi elementi costituiscono rumore antropico e NON DEVONO essere imitati nel tuo output. Utilizza questi esempi ESCLUSIVAMENTE per apprendere le logiche del bias. Nel tuo output, segui rigorosamente il formato JSON standard richiesto dalle istruzioni."""


# retry prompts
retry_spans_prompt_it = """Il tuo output precedente non era valido.
Devi restituire SOLO un singolo blocco <SPANS> in formato lista JSON.
Non scrivere altro testo. Riprova:


{instance}"""


retry_rationales_prompt_it = """Il tuo output precedente non era valido.
Devi restituire SOLO un singolo blocco <RATIONALES> in formato lista JSON.
Fornisci esattamente una razionale per ogni span, nello stesso ordine.
Non scrivere altro testo.
Se non ci sono span, restituisci esattamente <RATIONALES>: ["No"]. Riprova:


Estratto della notizia: {instance}
Span: {spans}"""


def mount_drive():
    drive.mount('/content/drive')


def load_api_key():
    api_key = userdata.get('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError('openrouter_api_key not found in colab secrets. please add it first.')
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

        return "FORMAT_ERROR: Empty field return from server connection."
    except Exception as e:
        return f"API_ERROR: {str(e)}"


# parsers stay the same
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


# build retrieval examples from the encoded bank
def build_dynamic_examples_string(query_emb, bank_embs, bank_df, k):
    search_results = util.semantic_search(query_emb, bank_embs, top_k=k)[0]
    examples_str = ""

    for i, result in enumerate(search_results):
        match_row = bank_df.iloc[result['corpus_id']]
        label = str(match_row['label_lower']).lower()
        spans = str(match_row['spans'])
        rats = str(match_row['rationales'])
        text = str(match_row['text'])

        examples_str += f"--- Esempio {i+1} ({label.upper()}) ---\n"
        examples_str += f"{text}\n"
        examples_str += f"<PREDICTED_LABEL>: {label}\n"
        examples_str += f"<SPANS>: {spans}\n"
        examples_str += f"<RATIONALES>: {rats}\n\n"

    return examples_str


# load previous output if it exists, otherwise start from the test file
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

    # retrieval first
    query_embedding = retriever_model.encode(
        article_text,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=device
    )
    a1_dynamic_examples = build_dynamic_examples_string(
        query_embedding,
        bank_embeddings,
        bank_df,
        K_EXAMPLES
    )

    # task 1: severity
    prompt_1 = f"{task_1_prompt_it}\n\n{a1_dynamic_examples}\n{global_override_it}\n\nOra elabora il seguente input:\n{article_text}"
    sev_raw = call_llm(prompt_1, system_prompt_it, 50, api_key)
    sev_parsed = parse_severity(sev_raw)

    # if severity is none, skip spans and rationales
    if sev_parsed == "none":
        print("severity is 'none', so spans and rationales are skipped")
        spans_raw = "SHORT_CIRCUITED_DUE_TO_NONE_SEVERITY"
        spans_parsed = '["No"]'
        rats_raw = "SHORT_CIRCUITED_DUE_TO_NONE_SEVERITY"
        rats_parsed = '["No"]'

    else:
        # task 2a: spans
        prompt_2a = f"{spans_task_prompt_it}\n\n{a1_dynamic_examples}\n{global_override_it}\n\nOra elabora il seguente input:\n{article_text}\n\nRestituisci l'output SOLO con un blocco <SPANS>. Non restituire blocchi multipli. Non scrivere span duplicati. Non aggiungere spiegazioni."
        spans_raw = call_llm(prompt_2a, system_prompt_it, 300, api_key)
        spans_parsed = parse_spans(spans_raw)

        if spans_parsed == "FORMAT_ERROR":
            print("span format error, trying one retry")
            retry_flag = True
            prompt_retry = retry_spans_prompt_it.format(instance=article_text)
            spans_raw_retry = call_llm(prompt_retry, system_prompt_it, 300, api_key)
            spans_raw = f"ATTEMPT 1:\n{spans_raw}\n\nATTEMPT 2:\n{spans_raw_retry}"
            spans_parsed = parse_spans(spans_raw_retry)
            if spans_parsed == "FORMAT_ERROR":
                spans_parsed = '["FORMAT_ERROR"]'
        elif spans_raw.startswith("API_ERROR"):
            spans_parsed = '["API_ERROR"]'

        # task 2b: rationales
        if spans_parsed in ['["API_ERROR"]', '["FORMAT_ERROR"]']:
            rats_raw = "SKIPPED_DUE_TO_SPANS_ERROR"
            rats_parsed = "SKIPPED_DUE_TO_SPANS_ERROR"
        else:
            # keep the maverick quote-fix because it is part of the original italian file
            try:
                temp_list = ast.literal_eval(spans_parsed)
                if isinstance(temp_list, list):
                    sanitized_spans = json.dumps(temp_list, ensure_ascii=False)
                else:
                    sanitized_spans = spans_parsed
            except Exception:
                sanitized_spans = spans_parsed.replace("'", '"')

            prompt_2b = f"{rationales_task_prompt_it}\n\n{a1_dynamic_examples}\n{global_override_it}\n\nOra elabora il seguente input:\nEstratto della notizia: {article_text}\nSpan: {sanitized_spans}"
            rats_raw = call_llm(prompt_2b, system_prompt_it, 800, api_key)
            rats_parsed = parse_rationales(rats_raw)

            if rats_parsed == "FORMAT_ERROR":
                print("rationale format error, trying one retry")
                retry_flag = True
                prompt_retry_rat = retry_rationales_prompt_it.format(instance=article_text, spans=sanitized_spans)
                rats_raw_retry = call_llm(prompt_retry_rat, system_prompt_it, 800, api_key)
                rats_raw = f"ATTEMPT 1:\n{rats_raw}\n\nATTEMPT 2:\n{rats_raw_retry}"
                rats_parsed = parse_rationales(rats_raw_retry)
                if rats_parsed == "FORMAT_ERROR":
                    print("rationale retry failed, forcing fallback")
                    rats_parsed = '["FORMAT_ERROR"]'
            elif rats_raw.startswith("API_ERROR") or rats_raw.startswith("FORMAT_ERROR"):
                rats_parsed = "FORMAT_ERROR"

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

    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"starting a1 italian run with model: {MODEL_NAME} [{DATA_LANGUAGE.upper()}]")
    print("loading bge-m3 retriever")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    retriever_model = SentenceTransformer('BAAI/bge-m3', device=device)

    print("loading exemplar bank")
    bank_df = pd.read_pickle(BANK_PKL)
    bank_embeddings = torch.from_numpy(np.array(bank_df['bge_m3_embedding'].tolist())).to(device)

    df = prepare_dataframe(TEST_CSV, OUTPUT_CSV)

    for index, row in df.iterrows():
        if str(row.get('severity_raw', "")).strip() != "":
            continue

        print(f"\nprocessing row {index + 1} with id: {row['text_id']}")
        result = process_row(row, api_key, retriever_model, device, bank_df, bank_embeddings)

        for key, value in result.items():
            df.at[index, key] = value

        df.to_csv(OUTPUT_CSV, index=False)
        time.sleep(ROW_SLEEP_SECONDS)

    print("a1 italian batch run complete")


if __name__ == '__main__':
    main()