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
# this file is the a1 farsi setup with retrieval plus target-language prompts
# if you switch model later, change the output name too so runs stay easy to track
DATA_LANGUAGE = "fa"
MODEL_NAME = "mistralai/mixtral-8x22b-instruct"
TEST_CSV = f'/your_path/{DATA_LANGUAGE}_pilot_test.csv'
BANK_PKL = f'/your_path/{DATA_LANGUAGE}_bge_m3_bank.pkl'
OUTPUT_CSV = f'/your_path/{DATA_LANGUAGE}_A1_mixtral_results.csv'
K_EXAMPLES = 4
REQUEST_TIMEOUT = 120
ROW_SLEEP_SECONDS = 1


system_prompt_fa = "شما یک هوش مصنوعی سخت‌گیر برای قالب‌بندی داده‌ها هستید. شما باید فقط تگ‌های درخواست‌شده را برگردانید. از نوشتن متن‌های محاوره‌ای، احوال‌پرسی یا توضیحات اضافی خودداری کنید."


# task 1: severity template
task_1_prompt_fa = """شما یک متخصص در حوزه چارچوب‌بندی، سوگیری زبانی و اختلال اطلاعات هستید.
اختلال اطلاعات به معنای آلودگی فضای اطلاعاتی است که شامل سه دسته اصلی می‌شود: کژاطلاعات (اطلاعات نادرست و سهوی)، دروغ‌رسانی (اطلاعات نادرست و عمدی) و اطلاعات مغرضانه (اطلاعات واقعی با هدف آسیب‌رسانی). این موارد اغلب با دستکاری احساسی، بسترسازی فریبنده یا چارچوب‌بندی تحریف‌شده مشخص می‌شوند.
وظیفه شما طبقه‌بندی شدت زبان مسئله‌دار در بخشی از یک مقاله خبری است.

شدت را بر اساس ویژگی‌های زیر طبقه‌بندی کنید:
- سوگیری رویدادی: توصیف رویدادها به‌گونه‌ای که مسئولیت را کم‌اهمیت جلوه دهد یا تحریف کند.
- سوگیری در توصیف افراد (سوگیری انتسابی): توصیف اشخاص، گروه‌ها یا جوامع با استفاده از زبانی احساسی، کلیشه‌ای، برچسب‌زننده یا اغراق‌آمیز.
- احساسات‌گرایی: بزرگ‌نمایی یا دراماتیزه‌کردن واقعیت‌ها.
- گمانه‌زنی‌ها: استفاده از عبارات مبهم به عنوان واقعیت.

برچسب‌های میزان شدت:
"none" – هیچ ویژگی مسئله‌داری وجود ندارد. متن واقع‌گرایانه، متعادل و خنثی است.
"slightly" – زبان مسئله‌دار جزئی وجود دارد، اما پیام کلی عمدتاً عینی باقی می‌ماند.
"moderately" – موارد متعددی از سوگیری یا زبان گمراه‌کننده وجود دارد که باعث تحریف ملموس شده و می‌تواند جهت‌گیری متن و درک مخاطب را تغییر دهد.
"highly" – استفاده گسترده و شدید از زبان سوگیرانه یا احساساتی. خطر بالای اطلاعات نادرست.

قالب خروجی:
فقط یکی از چهار برچسب را پس از تگ زیر برگردانید، دقیقاً به این شکل:
<PREDICTED_LABEL>: none
<PREDICTED_LABEL>: slightly
<PREDICTED_LABEL>: moderately
<PREDICTED_LABEL>: highly

هیچ توضیح اضافی یا متن دیگری اضافه نکنید. فقط برچسب معتبر را برگردانید.

{a1_dynamic_examples}

{global_override_fa}

اکنون ورودی زیر را پردازش کنید:
{instance}"""


# task 2: spans template
spans_task_prompt_fa = """شما یک متخصص در حوزه چارچوب‌بندی، سوگیری زبانی و اختلال اطلاعات هستید.
اختلال اطلاعات به معنای آلودگی فضای اطلاعاتی است که شامل سه دسته اصلی می‌شود: کژاطلاعات (اطلاعات نادرست و سهوی)، دروغ‌رسانی (اطلاعات نادرست و عمدی) و اطلاعات مغرضانه (اطلاعات واقعی با هدف آسیب‌رسانی).
این موارد اغلب با دستکاری احساسی، بسترسازی فریبنده یا چارچوب‌بندی تحریف‌شده مشخص می‌شوند.
وظیفه شما تجزیه و تحلیل گزیده‌های خبری و شناسایی بازه‌های متنی است که گمراه‌کننده، مغرضانه، سوداگرانه یا دارای بار احساسی هستند.

وظیفه:
- فقط بازه‌های منحصربه‌فرد و بدون همپوشانی را شناسایی کنید.
- اگر هیچ بازه‌ای یافت نشد، خروجی باید دقیقاً <SPANS>: ["No"] باشد.

قانون حفظ کاراکترها (بسیار مهم):
بازه‌های استخراج‌شده باید دقیقاً و کاراکتر به کاراکتر مطابق متن اصلی باشند.
به هیچ وجه املای کلمات، علائم نگارشی، فاصله‌ها و به ویژه نیم‌فاصله‌ها را تغییر ندهید.

بازه‌های مشکل‌ساز شامل:
- سوگیری رویدادی: رویدادها را به گونه‌ای توصیف می‌کند که مسئولیت تحریف شود.
- سوگیری در توصیف افراد (سوگیری انتسابی): توصیف اشخاص، گروه‌ها یا جوامع با استفاده از زبانی احساسی، کلیشه‌ای، برچسب‌زننده یا اغراق‌آمیز.
- واقعیت‌ها را هیجان‌انگیز یا اغراق‌آمیز جلوه می‌دهد.
- از اظهارات مبهم به گونه‌ای استفاده می‌کند که گویی واقعی هستند.

فرمت خروجی (دقیق):
اگر یک بازه: <SPANS>: ["..."]
اگر چند بازه: <SPANS>: ["...", "..."]
اگر هیچ بازه‌ای نیست: <SPANS>: ["No"]

{a1_dynamic_examples}

{global_override_fa}

اکنون ورودی زیر را پردازش کنید:
{instance}

پاسخ را فقط با یک بلوک <SPANS> برگردانید. بلوک‌های متعدد برنگردانید. بازه‌های تکراری ننویسید. هیچ توضیحی اضافه نکنید."""


# task 3: rationales template
rationales_task_prompt_fa = """شما یک متخصص در حوزه چارچوب‌بندی، سوگیری زبانی و اختلال اطلاعات هستید.
اختلال اطلاعات به معنای آلودگی فضای اطلاعاتی است که شامل سه دسته اصلی می‌شود: کژاطلاعات (اطلاعات نادرست و سهوی)، دروغ‌رسانی (اطلاعات نادرست و عمدی) و اطلاعات مغرضانه (اطلاعات واقعی با هدف آسیب‌رسانی).
این موارد اغلب با دستکاری احساسی، بسترسازی فریبنده یا چارچوب‌بندی تحریف‌شده مشخص می‌شوند.
وظیفه شما توضیح این است که چرا بازه‌های متنی خاص در یک خبر، گمراه‌کننده، مغرضانه یا مشکل‌ساز هستند.

به شما یک گزیده خبری و لیستی از بازه‌های استخراج‌شده داده می‌شود.
هدف شما تولید دقیقاً یک دلیل (rationale) برای هر بازه است.
از این قالب دقیق استفاده کنید:
"اگر [اشاره به بازه در متن]، آنگاه [پیامد یا نتیجه]"

دستورالعمل‌های خروجی (رعایت دقیق الزامی است):
- برای هر بازه دقیقاً یک دلیل به همان ترتیب برگردانید.
- اگر <SPANS>: ["No"] بود، شما نیز دقیقاً <RATIONALES>: ["No"] برگردانید. دلیل جدیدی نسازید.
- هر دلیل باید داخل گیومه (" ") باشد.
- همه دلایل را در یک لیست برگردانید، دقیقاً مطابق قالب نمونه زیر:
<RATIONALES>: ["اگر ..., آنگاه ...", "اگر ..., آنگاه ..."]
- از استفاده از نقل‌قول‌های تودرتو یا سه‌نقطه (...) در داخل دلایل خودداری کنید.
- بازه‌ها را ترکیب نکنید.
- دلیلی را جا نیندازید.
- مطلقاً هیچ فرآیند فکری، پیش‌نویس یا اصلاح خودکاری نداشته باشید.
- هیچ کلمه‌ای خارج از تگ‌ها ننویسید. فقط و فقط لیست نهایی را برگردانید.

{a1_dynamic_examples}

{global_override_fa}

اکنون ورودی زیر را پردازش کنید:
گزیده خبری: {instance}
بازه‌ها: {spans}"""


global_override_fa = """توجه مهم: مثال‌های ارائه‌شده مستقیماً از داده‌های انسانی استخراج شده‌اند و ممکن است بعضی از آنها دارای خطاهای نگارشی و قالب‌بندی باشند (مانند بازه‌های تکراری، هم‌خوانی نداشتن تعداد بازه‌ها و دلایل، عدم پیروی از قالب‌بندی پیشنهادی و وجود علائم اضافی مثل 0: و \\n). این موارد صرفاً خطای انسانی هستند و نباید در خروجی شما تکرار شوند. در صورت مشاهده، از این موارد فقط برای یادگیری منطق سوگیری استفاده کنید و در پاسخ خود، دقیقاً از قالب استاندارد درخواست‌شده پیروی کنید."""


# retry prompts
retry_spans_prompt_fa = """خروجی قبلی شما نامعتبر بود.
شما باید فقط و فقط یک بلوک <SPANS> با فرمت لیست JSON برگردانید.
هیچ متن دیگری ننویسید. دوباره تلاش کنید:

{instance}"""


retry_rationales_prompt_fa = """خروجی قبلی شما نامعتبر بود.
شما باید فقط و فقط یک بلوک <RATIONALES> با فرمت لیست JSON برگردانید.
دقیقاً یک دلیل برای هر بازه، به همان ترتیب قبلی، ارائه دهید.
هیچ متن دیگری ننویسید.
اگر هیچ بازه‌ای وجود نداشت، دقیقاً <RATIONALES>: ["No"] برگردانید. دوباره تلاش کنید:

گزیده خبری: {instance}
بازه‌ها: {spans}"""


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


# build retrieval examples dynamically from the encoded bank
def build_dynamic_examples_string(query_emb, bank_embs, bank_df, k):
    search_results = util.semantic_search(query_emb, bank_embs, top_k=k)[0]
    examples_str = ""

    for i, result in enumerate(search_results):
        match_row = bank_df.iloc[result['corpus_id']]
        label = str(match_row['label_lower']).lower()
        spans = str(match_row['spans'])
        rats = str(match_row['rationales'])
        text = str(match_row['text'])

        examples_str += f"--- مثال {i+1} ({label.upper()}) ---\n"
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


# process single row with bypass and retry
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

    # task 1: severity classification
    prompt_1 = task_1_prompt_fa.format(
        instance=article_text,
        a1_dynamic_examples=a1_dynamic_examples,
        global_override_fa=global_override_fa
    )
    sev_raw = call_llm(prompt_1, system_prompt_fa, 50, api_key)
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
        prompt_2a = spans_task_prompt_fa.format(
            instance=article_text,
            a1_dynamic_examples=a1_dynamic_examples,
            global_override_fa=global_override_fa
        )
        spans_raw = call_llm(prompt_2a, system_prompt_fa, 300, api_key)
        spans_parsed = parse_spans(spans_raw)

        if spans_parsed == "FORMAT_ERROR" and not spans_raw.startswith("API_ERROR"):
            print("span format invalid. triggering retry protocol...")
            retry_flag = True
            prompt_retry = retry_spans_prompt_fa.format(instance=article_text)
            spans_raw_retry = call_llm(prompt_retry, system_prompt_fa, 300, api_key)
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
            prompt_2b = rationales_task_prompt_fa.format(
                instance=article_text,
                spans=spans_parsed,
                a1_dynamic_examples=a1_dynamic_examples,
                global_override_fa=global_override_fa
            )
            rats_raw = call_llm(prompt_2b, system_prompt_fa, 800, api_key)
            rats_parsed = parse_rationales(rats_raw)

            if rats_parsed == "FORMAT_ERROR" and not rats_raw.startswith("API_ERROR"):
                print("rationale format invalid. triggering retry protocol...")
                retry_flag = True
                prompt_retry_rat = retry_rationales_prompt_fa.format(instance=article_text, spans=spans_parsed)
                rats_raw_retry = call_llm(prompt_retry_rat, system_prompt_fa, 800, api_key)
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

    print(f"starting a1 farsi run with model: {MODEL_NAME} [{DATA_LANGUAGE.upper()}]")

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
        if str(row.get('severity_raw', "")).strip() != "":
            continue

        print(f"\nprocessing row {index + 1} with id: {row['text_id']}")
        
        result = process_row(row, api_key, retriever_model, device, bank_df, bank_embeddings)

        for key, value in result.items():
            df.at[index, key] = value

        df.to_csv(OUTPUT_CSV, index=False)
        time.sleep(ROW_SLEEP_SECONDS)

    print("a1 farsi batch run complete")


if __name__ == '__main__':
    main()
