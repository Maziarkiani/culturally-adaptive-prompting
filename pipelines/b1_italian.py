import os
import re
import time
import pandas as pd
import requests
from google.colab import drive, userdata

# config:
# you can change the model to maverick or others if needed
# if you switch models later, update MODEL_NAME and OUTPUT_CSV to keep runs easy to track
MODEL_NAME = "meta-llama/llama-4-maverick"
LANGUAGE = "it"
INPUT_CSV = "it_pilot_test.csv"
OUTPUT_CSV = "it_B1_mixtral_results.csv"
REQUEST_TIMEOUT = 120
ROW_SLEEP_SECONDS = 1

# keep the system prompt strict so the model only returns the needed format
system_prompt_it = "Sei un'IA rigorosa per la formattazione dei dati. DEVI restituire SOLO i tag richiesti. Non includere riempitivi conversazionali, saluti o spiegazioni."

# task 1: severity classification template
task_1_prompt_few_shot_it = """Sei un esperto di framing, bias linguistici e Disordine dell'Informazione.
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

Non aggiungere spiegazioni, commenti o altro testo. Restituisci solo un'etichetta valida.

{b1_examples_it}

{global_override_it}

Ora processa il seguente input:
{instance}"""

# task 2: spans extraction template
spans_task_prompt_few_shot_it = """Sei un esperto di framing, bias linguistici e Disordine dell'Informazione.
Il disordine dell'informazione è l'inquinamento dello spazio informativo che include misinformazione, disinformazione e malinformazione. È spesso caratterizzato da manipolazione emotiva, falsi contesti o framing distorto.
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
Se non ci sono span: <SPANS>: ["No"]

{b1_examples_it}

{global_override_it}

Ora elabora il seguente input:
{instance}

Restituisci la risposta usando un solo blocco <SPANS>.
Non restituire più blocchi.
Non ripetere span duplicati.
Non aggiungere spiegazioni."""

# task 3: rationales generation template
rationales_task_prompt_few_shot_it = """Sei un esperto di framing, bias linguistici e Disordine dell'Informazione.
Il disordine dell'informazione è l'inquinamento dello spazio informativo che include misinformazione, disinformazione e malinformazione. È spesso caratterizzato da manipolazione emotiva, falsi contesti o framing distorto.
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
- Non scrivere alcuna parola fuori dai tag. Restituisci solo e soltanto la lista finale.

{b1_examples_it}

{global_override_it}

Ora elabora il seguente input:
Estratto di notizia: {instance}
Span: {spans}"""

# few-shot examples block matching original annotations
b1_examples_it = """
--- Esempio 0 (NONE) ---
TITLE: N/A
TITLE: La Regina Elisabetta governa personalmente l'Impero Britannico? Sì – RENOVATIO 21
SENTENCE_0: La mossa, scrive EIR,  costrinse un’elezione che gettò  il partido laburista australiano all’opposizione, concludendo quindi anche la politica di nazionalismo economico che Whitlam stava perseguendo, comprese le intenzioni di nazionalizzare le compagnie minerarie che avevano sfruttato le enormi risorse minerarie dell’Australia,  aziende considerate come i «gioielli della corona» della famiglia reale britannica che deteneva importanti partecipazioni in esse— ad esempio, la notissima Rio Tinto, compagnia di estrazione mineraria che nel 2009 finì accusata di spionaggio in Cina e che dovette chiedere la consulenza di Henry Kissinger per cercare di uscire dall’impasse.
SENTENCE_1: In una lettera del giorno successivo, Charteris fa un riferimento ancora più diretto all’utilizzo dei poteri di riserva, adducendo in particolare la preoccupazione di Kerr che qualsiasi decision presa potrebbe influenzare la monarchia: “Se fai, come vuoi, ciò che la Costituzione impone, non puoi assolutamente fare alla Monarchia alcun danno evitabile.
SENTENCE_2: Grazie alla determinazione della storica australiana Jenny Hocking, professore presso il National Center for Australian Studies, Monash University, è stata rivelata la prova del ruolo personale della regina, attraverso la corrispondenza appena rilasciata tra Kerr e la regina, attraverso il suo segretario personale,  Sir Martin Charteris.
<PREDICTED_LABEL>: none
<SPANS>: ["No"]
<RATIONALES>: ["No"]

--- Esempio 1 (SLIGHTLY) ---
TITLE: N/A
TITLE: «L'euro digitale è inevitabile». La BCE spinge verso la nostra schiavitù elettronica – RENOVATIO 21
 SENTENCE_0: L’euro digitale, ha dichiarato Panetta agirebbe «moneta sovrana utilizzabile da chiunque, famiglie, imprese commercianti», una valuta priva dei rischi «di mercato, credito e liquidità» in grado di tutelare la privacy – questo a differenza dai colossi privati Big Tech, i quali potrebbero mettere a a rischio la protezione di «aspetti provati della nostra vita o le tendenze politiche o sessuali o lo stato di salute» così come potrebbero interferire «con le regole di funzionamento di una moderne democrazia liberale».
SENTENCE_1: «Panetta, che è il responsabile del progetto lanciato ufficialmente nel luglio scorso dopo una lunga carriera in Banca d’Italia, ha sottolineato come 80 paesi nel mondo si stiano muovendo per la creazione di valute digitali mentre è sempre più forte la crescita delle criptovalute, degli stablecoin e delle transazioni di pagamento da parte dei grandi colossi tech» scrive l’ANSA, che nota come il messaggio vada «un po’ più al di là della tradizionale prudenza di Francoforte».
SENTENCE_2: Nell’ultima analisi della volatilità nei mercati delle materie prime, il responsabile della strategia sulle materie prime di TD Securities, Bart Melek, ha affermato che negli ultimi anni le banche centrali hanno registrato un aumento record degli acquisti di oro a causa delle preoccupazioni sulla sostenibilità del potere d’acquisto del dollaro e delle tensioni geopolitiche tra le principali potenze economiche.
<PREDICTED_LABEL>: slightly
<SPANS>: ['la nostra schiavitù elettronica']
<RATIONALES>: ['Se il titolo è posto in questi termini, allora potrebbe essere sensazionalistico e manipolare emotivamente i lettori']

--- Esempio 2 (MODERATELY) ---
TITLE: N/A
TITLE: Aiutare l’Africa nel solo modo possibile: andandocene via | LIBRE
SENTENCE_0: Voglio semplicemente dire che , dal momento che i rapaci sono sempre esistiti , il fenomeno di colonizzazione mascherato da aiuto umanitario ci sarà sempre  , ma un organizzazione per quanto ben congegnata e efficiente ha sempre bisogno di una base “volontaria” indottrinata , di persone che fanno dell’aiuto agli altri una ragione di vita , e in buona fede perchè non ci guadagnano nulla , sono anche “apprezzabili” da un certo punto di vista … ma prima avrebbero bisogno di terapie psicologiche per verificare i problemi di fondo che li portano a queste scelte e poi , chissà , potrebbero arrivare a capire che se non viene richiesto nessun aiuto non c’è nessun motivo di inventarsi problemi da riversare su chi si vorrebbe aiutare.
SENTENCE_1: Nei quattro anni del suo governo fece parecchie cose positive per il Burkina: si impegnò molto per eliminare la povertà attraverso il taglio degli sprechi statali e la soppressione dei privilegi delle classi agiate, finanziò un ampio sistema di riforme sociali incentrato sulla costruzione di scuole, ospedali e case per la popolazione estremamente povera, fece un’importante lotta alla desertificazione con il piantamento di milioni di alberi nel Sahel, cercò di svincolare il paese dalle importazioni forzate.
SENTENCE_2: Posco immaginare perfino sul piano linguistico: poiché col “reddito di cittadinanza” alle generazioni nuove del Sud non occorrerà più studiare né imparare a leggere far di conto, parleranno in modo sempre più esclusivo i vernacoli sempre più gutturali e belluini; nella decrescita felice andranno alla raccolta dei mitili e  bacche e impareranno ad abitare sulle  palafitte (abusive, naturalmente), come  i cacciatori-raccoglitori della civiltà  villanoviana.
<PREDICTED_LABEL>: moderately
<SPANS>: ['i vernacoli sempre più gutturali e belluini', 'nella decrescita felice andranno alla raccolta dei mitili e bacche e impareranno ad abitare sulle palafitte (abusive, naturalmente), come i cacciatori-raccoglitori della civiltà villanoviana']
<RATIONALES>: ['attribuire alle lingue locali gli aggettivi "gutturali" e "belluini" è offensivo', 'si tratta di un assunto per cui i popoli africani siano da considerare selvaggi e arretrati']

--- Esempio 3 (HIGHLY) ---
TITLE: N/A
TITLE: Vaccino e aborto, Pfizer ha cercato di nascondere l'uso di cellule fetali: email trapelate – RENOVATIO 21
 SENTENCE_0: L’idea che Renovatio 21 portava avanti, in reality ancora prima dei tempi della legge Lorenzin (2017) che impediva l’accesso alle scuole ai nostri figli, era semplice: sollevando il tema dei feti sacrificati dei vaccini, ci sarà possibile utilizzare l’obiezione di coscienza – grande vacca sacra apparentemente intoccabile così come la sua legge genocidiaria 194/78, perché teneva in piedi la pax bioetica assassina del compromesso tra democristiani, comunisti e radicali (e, oggi, i loro figli postcomunisti, postfascisti, «liberali» – in company ovviamente dei vescovi) – e così sottrarre milioni di bambini all’abominio della siringa di Stato obbligatoria, contenente – come nelle antiche pozioni delle streghe descritte dal Malleus maleficarum (1486)– pezzi di bimbo sacrificato.
SENTENCE_1: La densa conversazione, della durata di quasi due ore, è stata pubblicata su X il 27 dicembre e ha attirato grande attenzione sui social media, poiché il Siri, che sta aiutando Kennedy a esaminare i potenziali dipendenti del dipartimento della Salute e dei Servizi Umani, è stato oggetto di un feroce articolo sul New York Times il mese scorso che lo accusava di aver intentato una causa mettere al bando il vaccino contro la poliomielite.
SENTENCE_2: Non fu così: non solo la gerarchia cattolica della Roma conciliare corrotta, simoniaca e demoniaca aveva pronte giustificazioni e scuse fatte di pura menzogna: no, di lì a poco sarebbe scattata la più grande emergenza sperimentata dall’umanità del XXI secolo, la pandemia, che, guarda guarda, nella distruzione totale dei diritti costituzionali dei cittadini ad un certo punto richiese proprio il vaccino obbligatorio per l’intera popolazione.
<PREDICTED_LABEL>: highly
<SPANS>: ["Vaccino e aborto, Pfizer ha cercato di nascondere l'uso di cellule fetali", 'feti sacrificati dei vaccini', 'feti sacrificati dei vaccini']
<RATIONALES>: ["se utilizzi un'espressione molto provocatoria, allora vuoi creare una visione distorta del tema", 'se definisci la vaccinazione come abominio che è un termine non neutro, allora vuoi polarizzare il pubblico ', "se utilizzi un'espressione molto provocatoria, allora vuoi creare una visione distorta del tema", 'se parli del tentativo intenzionale di Pfizer senza fornire prove concrete, allora alimenti teorie del complunto e paure infondate', "se utilizzi un'espressione molto forte ed emotiva, allora vuoi polarizzare il discorso pubblico e crea una visione distorta del tema"]
"""

# guideline for human annotations noise
global_override_it = """Attenzione importante: gli esempi forniti sono estratti direttamente da annotazioni umane e alcuni possono contenere errori di scrittura e di formattazione (come span duplicati, mancata corrispondenza tra numero di span e razionali, non rispettando il formato proposto, oppure segni extra come TITLE: ripetuto, \xa0 o altri artefatti). Questi elementi sono solo rumore umano e NON devono essere imitati nel tuo output. Usa gli esempi solo per apprendere la logica del bias e nella tua risposta segui esattamente il formato standard richiesto."""

# retry prompts
retry_spans_prompt_it = """Il tuo output precedente non era valido.
Devi restituire solo e soltanto un unico blocco <SPANS> in formato lista JSON.
Non scrivere nessun altro testo. Riprova:

{instance}"""

retry_rationales_prompt_it = """Il tuo output precedente non era valido.
Devi restituire solo e soltanto un unico blocco <RATIONALES> in formato lista JSON.
Restituisci esattamente UNA razionale per ogni span, nello stesso ordine.
Non scrivere nessun altro testo.
Se non era presente alcuno span, restituisci esattamente <RATIONALES>: ["No"]. Riprova:

Estratto di notizia: {instance}
Span: {spans}"""


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


# process single row with bypass logic and retry handlers
def process_row(row, api_key):
    article_text = str(row['text'])
    retry_flag = False

    # task 1: severity classification (few-shot template injection)
    prompt_1 = task_1_prompt_few_shot_it.format(
        instance=article_text,
        b1_examples_it=b1_examples_it,
        global_override_it=global_override_it
    )
    sev_raw = call_llm(prompt_1, system_prompt_it, 50, api_key)
    sev_parsed = parse_severity(sev_raw)

    # downward short-circuit for "none" severity cases
    if sev_parsed == "none":
        print("severity evaluated as 'none'. short-circuiting downstream layers.")
        spans_raw = "SHORT_CIRCUITED_DUE_TO_NONE_SEVERITY"
        spans_parsed = '["No"]'
        rats_raw = "SHORT_CIRCUITED_DUE_TO_NONE_SEVERITY"
        rats_parsed = '["No"]'
    else:
        # task 2a: spans extraction
        prompt_2a = spans_task_prompt_few_shot_it.format(
            instance=article_text,
            b1_examples_it=b1_examples_it,
            global_override_it=global_override_it
        )
        spans_raw = call_llm(prompt_2a, system_prompt_it, 300, api_key)
        spans_parsed = parse_spans(spans_raw)

        if spans_parsed == "FORMAT_ERROR" and not spans_raw.startswith("API_ERROR"):
            print("span format invalid. triggering retry protocol...")
            retry_flag = True
            prompt_retry = retry_spans_prompt_it.format(instance=article_text)
            spans_raw_retry = call_llm(prompt_retry, system_prompt_it, 300, api_key)
            spans_raw = f"ATTEMPT 1:\n{spans_raw}\n\nATTEMPT 2:\n{spans_raw_retry}"
            spans_parsed = parse_spans(spans_raw_retry)
            if spans_parsed == "FORMAT_ERROR":
                spans_parsed = '["FORMAT_ERROR"]'
            elif spans_raw.startswith("API_ERROR"):
                spans_parsed = '["API_ERROR"]'
        elif spans_raw.startswith("API_ERROR"):
            spans_parsed = '["API_ERROR"]'

        # task 2b: rationales generation
        if spans_parsed in ['["API_ERROR"]', '["FORMAT_ERROR"]']:
            rats_raw = "SKIPPED_DUE_TO_SPANS_ERROR"
            rats_parsed = "SKIPPED_DUE_TO_SPANS_ERROR"
        else:
            prompt_2b = rationales_task_prompt_few_shot_it.format(
                instance=article_text,
                spans=spans_parsed,
                b1_examples_it=b1_examples_it,
                global_override_it=global_override_it
            )
            rats_raw = call_llm(prompt_2b, system_prompt_it, 800, api_key)
            rats_parsed = parse_rationales(rats_raw)

            if rats_parsed == "FORMAT_ERROR" and not rats_raw.startswith("API_ERROR"):
                print("rationale format invalid. triggering retry protocol...")
                retry_flag = True
                prompt_retry_rat = retry_rationales_prompt_it.format(instance=article_text, spans=spans_parsed)
                rats_raw_retry = call_llm(prompt_retry_rat, system_prompt_it, 800, api_key)
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

    print(f"starting italian b1 run with model: {MODEL_NAME} [{LANGUAGE.upper()}]")
    print(f"input file: {INPUT_CSV}")
    print(f"output file: {OUTPUT_CSV}")

    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df = prepare_dataframe(INPUT_CSV, OUTPUT_CSV)

    for index, row in df.iterrows():
        # skip rows that have already been evaluated successfully
        if pd.notna(row.get('severity_raw', "")) and str(row.get('severity_raw', "")).strip() != "":
            continue

        print(f"\nprocessing row {index + 1} with id: {row['text_id']}")
        result = process_row(row, api_key)

        for key, value in result.items():
            df.at[index, key] = value

        df.to_csv(OUTPUT_CSV, index=False)
        time.sleep(ROW_SLEEP_SECONDS)

    print("italian b1 batch run complete.")


if __name__ == '__main__':
    main()
