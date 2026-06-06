import os
import re
import time
import pandas as pd
import requests
from google.colab import drive, userdata


# you can switch this to mixtral, or any other model later. Fix the path. 
MODEL_NAME = 'meta-llama/llama-4-maverick'
LANGUAGE = 'it'
INPUT_CSV = 'it_pilot_test.csv'
OUTPUT_CSV = 'it_B0_results.csv'
REQUEST_TIMEOUT = 120
ROW_SLEEP_SECONDS = 1


# keep the system prompt strict so the model returns only the tags we need
SYSTEM_PROMPT_IT = "Sei un'IA rigorosa per la formattazione dei dati. DEVI restituire SOLO i tag richiesti. Non includere riempitivi conversazionali, saluti o spiegazioni."


# task 1: first decide how problematic the language is
TASK_1_PROMPT_ZERO_SHOT_IT = """Sei un esperto di framing, bias linguistici e Disordine dell'Informazione.
Il disordine dell'informazione è l'inquinamento dello spazio informativo che include misinformazione, disinformazione e malinformazione. È spesso caratterizzato da manipolazione emotiva, falsi contesti o framing distorto.
Il tuo compito è classificare la gravità del linguaggio problematico in un estratto di un articolo di notizie.

Classifica la gravità in base alla presenza e all’impatto delle seguenti caratteristiche problematiche:
- Bias eventivo: descrive o fa riferimento a eventi in un modo che minimizza o distorce la responsabilità.
- Bias attributivo: descrive persone, groups o comunità usando un linguaggio emotivamente carico, stereotipato, etichettante o esagerato.
- Sensazionalismo: esagera o drammatizza i fatti.
- Speculazione: usa affermazioni vaghe o speculative como se fossero fattuali.

ETICHETTE DI GRAVITÀ:
"none" – Nessuna caratteristica problematica presente. Il testo è fattuale, equilibrato e neutrale.
"slightly" – È presente un linguaggio leggermente problematico, ma il messaggio complessivo rimane per lo più fattuale.
"moderately" – Sono presenti più casi di bias o linguaggio fuorviante che producono una distorsione tangibile e possono alterare la comprensione del lettore.
"highly" – Uso grave e pervasivo di linguaggio parziale, speculativo o emotivamente manipolativo. Alto rischio di disinformazione.

FORMATO DI OUTPUT:
Restituisci solo una delle quattro etichette dopo il tag seguente, esattamente in questo modo:
: none
: slightly
: moderately
: highly

Non aggiungere spiegazioni, commenti o altro testo. Restituisci solo un'etichetta valida.

Ora processa il seguente input:
{instance}"""


# task 2: only if needed, find the problematic spans in the article
SPANS_TASK_PROMPT_ZERO_SHOT_IT = """Sei un esperto di framing, bias linguistici e Disordine dell'Informazione.
Il disordine dell'informazione è l'inquinamento dello spazio informativo che include misinformazione, disinformazione e malinformazione. È spesso caratterizzato da manipolazione emotiva, falsi contesti o framing distorto.
Il tuo compito è analizzare estratti di notizie e identificare span di testo che sono fuorvianti, di parte, speculativi o emotivamente caricati.

Compito:
- Identifica solo span unici e NON sovrapposti.
- Se non trovi alcuno span problematico, l'output deve essere esattamente : ["No"].

Regola di conservazione dei caratteri (molto importante):
Gli span estratti devono corrispondere esattamente, carattere per carattere, al testo originale.
Non modificare in alcun modo ortografia, punteggiatura, maiuscole, apostrofi, accenti o spazi.

Gli span problematici includono:
- Bias eventivo: gli eventi sono descritti in modo da minimizzare o distorcere la responsabilità.
- Bias attributivo: persone, gruppi o comunità sono descritti con linguaggio emotivamente carico, stereotipato, etichettante o esagerato.
- Il testo sensazionalizza o drammatizza i fatti.
- Il testo usa affermazioni vaghe o speculative come se fossero fattuali.

FORMATO DI OUTPUT (rigido):
Se c'è uno span: : ["..."]
Se ci sono più span: : ["...", "..."]
Se non ci sono span: : ["No"]

Ora elabora il seguente input:
{instance}

Restituisci la risposta usando un solo blocco .
Non restituire più blocchi.
Non ripetere span duplicati.
Non aggiungere spiegazioni."""


# task 3: explain why each extracted span is problematic
RATIONALES_TASK_PROMPT_ZERO_SHOT_IT = """Sei un esperto di framing, bias linguistici e Disordine dell'Informazione.
Il disordine dell'informazione è l'inquinamento dello spazio informativo che include misinformazione, disinformazione e malinformazione. È spesso caratterizzato da manipolazione emotiva, falsi contesti o framing distorto.
Il tuo compito è spiegare perché determinati span di testo in un estratto di notizie sono fuorvianti, di parte o problematici.

Ti vengono forniti un estratto di notizia e un elenco di span estratti.
Il tuo obiettivo è generare esattamente UNA razionale per ogni span.
Usa rigorosamente questo formato:
"se [riferimento allo span nel testo], allora [implicazione o conseguenza]"

Istruzioni di output (obbligatorie):
- Restituisci esattamente una razionale per ogni span, nello stesso ordine.
- Se : ["No"], allora devi restituire esattamente : ["No"]. Non inventare razionali.
- Ogni razionale deve essere racchiusa tra virgolette doppie (" ").
- Tutte le razionali devono essere restituite in un'unica lista:
: ["se ..., allora ...", "se ..., allora ..."]
- Non usare virgolette annidate o puntini di sospensione (...) all'interno delle rationales.
- Non combinare più span in una sola razionale.
- Non saltare span.
- Non mostrare processo di pensiero, bozze o auto-correzioni.
- Non scrivere alcuna parola fuori dai tag. Restituisci solo e soltanto la lista finale.

Ora elabora il seguente input:
Estratto di notizia: {instance}
Span: {spans}"""


# if the model breaks the format once, give it one clean retry
RETRY_SPANS_PROMPT_IT = """Il tuo output precedente non era valido.
Devi restituire solo e soltanto un unico blocco in formato lista JSON.
Non scrivere nessun altro testo. Riprova:

{instance}"""


RETRY_RATIONALES_PROMPT_IT = """Il tuo output precedente non era valido.
Devi restituire solo e soltanto un unico blocco in formato lista JSON.
Restituisci esattamente UNA razionale per ogni span, nello stesso ordine.
Non scrivere nessun altro testo.
Se non era presente alcuno span, restituire esattamente : ["No"]. Riprova:

Estratto di notizia: {instance}
Span: {spans}"""


def load_api_key():
    api_key = userdata.get('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError('openrouter_api_key not found in colab secrets. please add it first.')
    return api_key


# this is separate so you can change mount behavior later if needed
def mount_drive():
    drive.mount('/content/drive')


# keep the api call simple and readable
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
            message_obj = data['choices'][0].get('message', {})
            content_obj = message_obj.get('content', '')
            if content_obj is not None:
                return str(content_obj).strip()

        return 'FORMAT_ERROR: Empty response field.'
    except Exception as e:
        return f'API_ERROR: {str(e)}'


# task 1 parser: read the predicted severity label
def parse_severity(text):
    if text.startswith('API_ERROR') or text.startswith('FORMAT_ERROR'):
        return 'API_ERROR'

    match = re.search(r':\s*(none|slightly|moderately|highly)', text, re.IGNORECASE)
    return match.group(1).lower() if match else 'FORMAT_ERROR'


# task 2 parser: keep the span list after the tag
def parse_spans(text):
    if text.startswith('API_ERROR') or text.startswith('FORMAT_ERROR'):
        return 'API_ERROR'

    match = re.search(r':\s*(\[[\s\S]*\])', text)
    return match.group(1).strip() if match else 'FORMAT_ERROR'


# task 3 parser: same idea, but for rationales
def parse_rationales(text):
    if text.startswith('API_ERROR') or text.startswith('FORMAT_ERROR'):
        return 'API_ERROR'

    match = re.search(r':\s*(\[[\s\S]*\])', text)
    return match.group(1).strip() if match else 'FORMAT_ERROR'


# these are the columns we want to save back into the result file
def prepare_dataframe(input_csv):
    df = pd.read_csv(input_csv, dtype=str).fillna('')

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


# one row goes through severity first, then spans, then rationales
def process_row(row, api_key):
    article_text = str(row['text'])
    retry_flag = False

    prompt_1 = TASK_1_PROMPT_ZERO_SHOT_IT.format(instance=article_text)
    sev_raw = call_llm(prompt_1, SYSTEM_PROMPT_IT, 50, api_key)
    sev_parsed = parse_severity(sev_raw)

    if sev_parsed == 'none':
        print("severity is 'none', so spans and rationales are skipped")
        spans_raw = 'SKIPPED_DUE_TO_NONE_SEVERITY'
        spans_parsed = '["No"]'
        rats_raw = 'SKIPPED_DUE_TO_NONE_SEVERITY'
        rats_parsed = '["No"]'
    else:
        prompt_2a = SPANS_TASK_PROMPT_ZERO_SHOT_IT.format(instance=article_text)
        spans_raw = call_llm(prompt_2a, SYSTEM_PROMPT_IT, 300, api_key)
        spans_parsed = parse_spans(spans_raw)

        if spans_parsed == 'FORMAT_ERROR':
            print('span format error, trying one retry')
            retry_flag = True
            prompt_retry = RETRY_SPANS_PROMPT_IT.format(instance=article_text)
            spans_raw_retry = call_llm(prompt_retry, SYSTEM_PROMPT_IT, 300, api_key)
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
            prompt_2b = RATIONALES_TASK_PROMPT_ZERO_SHOT_IT.format(
                instance=article_text,
                spans=spans_parsed
            )
            rats_raw = call_llm(prompt_2b, SYSTEM_PROMPT_IT, 800, api_key)
            rats_parsed = parse_rationales(rats_raw)

            if rats_parsed == 'FORMAT_ERROR':
                print('rationale format error, trying one retry')
                retry_flag = True
                prompt_retry_rat = RETRY_RATIONALES_PROMPT_IT.format(
                    instance=article_text,
                    spans=spans_parsed
                )
                rats_raw_retry = call_llm(prompt_retry_rat, SYSTEM_PROMPT_IT, 800, api_key)
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


# main run: load data, process row by row, save after each row
def main():
    mount_drive()
    api_key = load_api_key()

    print(f'starting italian b0 run with model: {MODEL_NAME}')
    print(f'input file: {INPUT_CSV}')
    print(f'output file: {OUTPUT_CSV}')

    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    df = prepare_dataframe(INPUT_CSV)

    for index, row in df.iterrows():
        print(f"\nprocessing row {index + 1} with id: {row['text_id']}")
        result = process_row(row, api_key)

        for key, value in result.items():
            df.at[index, key] = value

        df.to_csv(OUTPUT_CSV, index=False)
        time.sleep(ROW_SLEEP_SECONDS)

    print('italian b0 batch run complete')


if __name__ == '__main__':
    main()