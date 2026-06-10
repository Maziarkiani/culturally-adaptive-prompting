import os
import re
import time
import pandas as pd
import requests
from google.colab import drive, userdata

# config:
# you can change the model to mixtral or others if needed
# if you switch models later, update MODEL_NAME and OUTPUT_CSV to keep runs easy to track
MODEL_NAME = "meta-llama/llama-4-maverick"
LANGUAGE = "fa"
INPUT_CSV = "fa_pilot_test.csv"
OUTPUT_CSV = "fa_B0_maverick_results.csv"
REQUEST_TIMEOUT = 120
ROW_SLEEP_SECONDS = 1

# keep the system prompt strict so the model only returns the needed format
system_prompt_fa = "شما یک هوش مصنوعی سخت‌گیر برای قالب‌بندی داده‌ها هستید. شما باید فقط تگ‌های درخواست‌شده را برگردانید. از نوشتن متن‌های محاوره‌ای، احوال‌پرسی یا توضیحات اضافی خودداری کنید."

# task 1: severity classification (original tags fully preserved)
task_1_prompt_zero_shot_fa = """شما یک متخصص در حوزه چارچوب‌بندی، سوگیری زبانی و اختلال اطلاعات هستید.
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

اکنون ورودی زیر را پردازش کنید:
{instance}"""

# task 2: spans extraction
spans_task_prompt_zero_shot_fa = """شما یک متخصص در حوزه چارچوب‌بندی، سوگیری زبانی و اختلال اطلاعات هستید.
اختلال اطلاعات به معنای آلودگی فضای اطلاعاتی است که شامل سه دسته اصلی می‌شود: کژاطلاعات (اطلاعات نادرست و سهوی)، دروغ‌رسانی (اطلاعات نادرست و عمدی) و اطلاعات مغرضانه (اطلاعات واقعی با هدف آسیب‌رسانی). این موارد اغلب با دستکاری احساسی، بسترسازی فریبنده یا چارچوب‌بندی تحریف‌شده مشخص می‌شوند.
وظیفه شما تجزیه و تحلیل گزیده‌های خبری و شناسایی بازه‌های متنی است که گمراه‌کننده، مغرضانه، سوداگرانه یا دارای بار احساسی هستند.

وظیفه:
- فقط بازه‌های منحصربه‌فرد و بدون همپوشانی را شناسایی کنید.
- اگر هیچ بازه‌ای یافت نشد، خروجی باید دقیقاً <SPANS>: ["No"] باشد.

قانون حفظ کاراکترها:
بازه‌های استخراج‌شده باید دقیقاً و کاراکتر به کاراکتر مطابق متن اصلی باشند. به هیچ وجه املای کلمات، علائم نگارشی، فاصله‌ها و به ویژه نیم‌فاصله‌ها را تغییر ندهید.

بازه‌های مشکل‌ساز شامل:
- سوگیری رویدادی: رویدادها را به گونه‌ای توصیف می‌کند که مسئولیت تحریف شود.
- سوگیری در توصیف افراد (سوگیری انتسابی): توصیف اشخاص، گروه‌ها یا جوامع با استفاده از زبانی احساسی، کلیشه‌ای، برچسب‌زننده یا اغراق‌آمیز.
- واقعیت‌ها را هیجان‌انگیز یا اغراق‌آمیز جلوه می‌دهد.
- از اظهارات مبهم به گونه‌ای استفاده می‌کند که گویی واقعی هستند.

فرمت خروجی:
اگر یک بازه: <SPANS>: ["..."]
اگر چند بازه: <SPANS>: ["...", "..."]
اگر هیچ بازه‌ای نیست: <SPANS>: ["No"]

اکنون ورودی زیر را پردازش کنید:
{instance}

پاسخ را فقط با یک بلوک <SPANS> برگردانید. بلوک‌های متعدد برنگردانید. بازه‌های تکراری ننویسید. هیچ توضیحی اضافه نکنید."""

# task 3: rationales generation
rationales_task_prompt_zero_shot_fa = """شما یک متخصص در حوزه چارچوب‌بندی، سوگیری زبانی و اختلال اطلاعات هستید.
اختلال اطلاعات به معنای آلودگی فضای اطلاعاتی است که شامل سه دسته اصلی می‌شود: کژاطلاعات (اطلاعات نادرست و سهوی)، دروغ‌رسانی (اطلاعات نادرست و عمدی) و اطلاعات مغرضانه (اطلاعات واقعی با هدف آسیب‌رسانی). این موارد اغلب با دستکاری احساسی، بسترسازی فریبنده یا چارچوب‌بندی تحریف‌شده مشخص می‌شوند.
وظیفه شما توضیح این است که چرا بازه‌های متنی خاص در یک خبر، گمراه‌کننده، مغرضانه یا مشکل‌ساز هستند.

به شما یک گزیده خبری و لیستی از بازه‌های استخراج‌شده داده می‌شود.
هدف شما تولید دقیقاً یک دلیل (rationale) برای هر بازه است.
از این قالب دقیق استفاده کنید: "اگر [اشاره به بازه در متن]، آنگاه [پیامد یا نتیجه]"

دستورالعمل‌های خروجی:
- برای هر بازه دقیقاً یک دلیل به همان ترتیب برگردانید.
- اگر <SPANS>: ["No"] بود، شما نیز دقیقاً <RATIONALES>: ["No"] برگردانید. دلیل جدیدی نسازید.
- هر دلیل باید داخل گیومه (" ") باشد.
- همه دلایل را در یک لیست برگردانید: <RATIONALES>: ["اگر ...، آنگاه ...", "اگر ...، آنگاه ..."]
- از استفاده از نقل‌قول‌های تودرتو یا سه‌نقطه در داخل دلایل خودداری کنید.
- بازه‌ها را ترکیب نکنید.
- هیچ متن دیگری خارج از تگ‌ها ننویسید.

اکنون ورودی زیر را پردازش کنید:
گزیده خبری: {instance}
بازه‌ها: {spans}"""

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


# process single row with short-circuit and retries
def process_row(row, api_key):
    article_text = str(row['text'])
    retry_flag = False

    # task 1: severity
    prompt_1 = task_1_prompt_zero_shot_fa.format(instance=article_text)
    sev_raw = call_llm(prompt_1, system_prompt_fa, 50, api_key)
    sev_parsed = parse_severity(sev_raw)

    # downward short-circuit for "none"
    if sev_parsed == "none":
        print("severity evaluated as 'none'. short-circuiting downstream layers.")
        spans_raw = "SHORT_CIRCUITED_DUE_TO_NONE_SEVERITY"
        spans_parsed = '["No"]'
        rats_raw = "SHORT_CIRCUITED_DUE_TO_NONE_SEVERITY"
        rats_parsed = '["No"]'
    else:
        # task 2a: spans
        prompt_2a = spans_task_prompt_zero_shot_fa.format(instance=article_text)
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

        # task 2b: rationales
        if spans_parsed in ['["API_ERROR"]', '["FORMAT_ERROR"]']:
            rats_raw = "SKIPPED_DUE_TO_SPANS_ERROR"
            rats_parsed = "SKIPPED_DUE_TO_SPANS_ERROR"
        else:
            prompt_2b = rationales_task_prompt_zero_shot_fa.format(instance=article_text, spans=spans_parsed)
            rats_raw = call_llm(prompt_2b, system_prompt_fa, 800, api_key)
            rats_parsed = parse_rationales(rats_raw)

            if rats_parsed == "FORMAT_ERROR":
                print("rationale format invalid. triggering retry protocol...")
                retry_flag = True
                prompt_retry_rat = retry_rationales_prompt_fa.format(instance=article_text, spans=spans_parsed)
                rats_raw_retry = call_llm(prompt_retry_rat, system_prompt_fa, 800, api_key)
                rats_raw = f"ATTEMPT 1:\n{rats_raw}\n\nATTEMPT 2:\n{rats_raw_retry}"
                rats_parsed = parse_rationales(rats_raw_retry)
                if rats_parsed == "FORMAT_ERROR":
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

    print(f"starting farsi b0 run with model: {MODEL_NAME} [{LANGUAGE.upper()}]")
    print(f"input file: {INPUT_CSV}")
    print(f"output file: {OUTPUT_CSV}")

    df = prepare_dataframe(INPUT_CSV, OUTPUT_CSV)

    for index, row in df.iterrows():
        # skip already processed rows safely
        if pd.notna(row.get('severity_raw', "")) and str(row.get('severity_raw', "")).strip() != "":
            continue

        print(f"\nprocessing row {index + 1} with id: {row['text_id']}")
        result = process_row(row, api_key)

        for key, value in result.items():
            df.at[index, key] = value

        df.to_csv(OUTPUT_CSV, index=False)
        time.sleep(ROW_SLEEP_SECONDS)

    print("farsi b0 batch run complete.")


if __name__ == '__main__':
    main()
