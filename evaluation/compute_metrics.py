# Evaluation script for the Culturally Adaptive Retrieval-Based Framework
# Computes severity classification (macro F1), span detection (overlap F1),
# and rationale quality (BERTScore F1) across all conditions, models, and languages.
# See thesis Chapter 6 for full description of evaluation methodology.
#
# Requirements: pip install pandas numpy scikit-learn evaluate matplotlib seaborn
# Set BASE_DIR to the folder containing your pipeline result CSVs before running.

import os
import ast
import numpy as np
import pandas as pd
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

BASE_DIR   = './results'   # folder containing the pipeline output CSVs
OUTPUT_DIR = './output'    # folder where the figure will be saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

# registry of all 16 conditions (4 setups x 2 models x 2 languages)
FILES = [
    {"file": "fa_B0_maverick_results.csv", "setup": "B0 (Zero-Shot)",   "lang": "Farsi",   "model": "Maverick", "lang_code": "fa"},
    {"file": "fa_B0_mixtral_results.csv",  "setup": "B0 (Zero-Shot)",   "lang": "Farsi",   "model": "Mixtral",  "lang_code": "fa"},
    {"file": "fa_B1_maverick_results.csv", "setup": "B1 (Static)",      "lang": "Farsi",   "model": "Maverick", "lang_code": "fa"},
    {"file": "fa_B1_mixtral_results.csv",  "setup": "B1 (Static)",      "lang": "Farsi",   "model": "Mixtral",  "lang_code": "fa"},
    {"file": "fa_M1_maverick_results.csv", "setup": "M1 (Dynamic)",     "lang": "Farsi",   "model": "Maverick", "lang_code": "fa"},
    {"file": "fa_M1_mixtral_results.csv",  "setup": "M1 (Dynamic)",     "lang": "Farsi",   "model": "Mixtral",  "lang_code": "fa"},
    {"file": "fa_A1_maverick_results.csv", "setup": "A1 (Target-Inst)", "lang": "Farsi",   "model": "Maverick", "lang_code": "fa"},
    {"file": "fa_A1_mixtral_results.csv",  "setup": "A1 (Target-Inst)", "lang": "Farsi",   "model": "Mixtral",  "lang_code": "fa"},
    {"file": "it_B0_maverick_results.csv", "setup": "B0 (Zero-Shot)",   "lang": "Italian", "model": "Maverick", "lang_code": "it"},
    {"file": "it_B0_mixtral_results.csv",  "setup": "B0 (Zero-Shot)",   "lang": "Italian", "model": "Mixtral",  "lang_code": "it"},
    {"file": "it_B1_maverick_results.csv", "setup": "B1 (Static)",      "lang": "Italian", "model": "Maverick", "lang_code": "it"},
    {"file": "it_B1_mixtral_results.csv",  "setup": "B1 (Static)",      "lang": "Italian", "model": "Mixtral",  "lang_code": "it"},
    {"file": "it_M1_maverick_results.csv", "setup": "M1 (Dynamic)",     "lang": "Italian", "model": "Maverick", "lang_code": "it"},
    {"file": "it_M1_mixtral_results.csv",  "setup": "M1 (Dynamic)",     "lang": "Italian", "model": "Mixtral",  "lang_code": "it"},
    {"file": "it_A1_maverick_results.csv", "setup": "A1 (Target-Inst)", "lang": "Italian", "model": "Maverick", "lang_code": "it"},
    {"file": "it_A1_mixtral_results.csv",  "setup": "A1 (Target-Inst)", "lang": "Italian", "model": "Mixtral",  "lang_code": "it"},
]

# pipeline error tokens excluded from evaluation
# items with these values in any column are dropped before scoring
ERROR_TOKENS = [
    'FORMAT_ERROR', 'API_ERROR', 'EMPTY_RESPONSE', 'SAFETY_REFUSAL', 'PROVIDER_ERROR',
    'SKIPPED_DUE_TO_SPANS_ERROR', '["FORMAT_ERROR"]', '["API_ERROR"]', '["EMPTY_RESPONSE"]',
    'HALTED_DUE_TO_SEVERITY_API_ERROR', 'HALTED_DUE_TO_SEVERITY_EMPTY_RESPONSE',
    'HALTED_DUE_TO_SEVERITY_SAFETY_REFUSAL', 'HALTED_DUE_TO_SEVERITY_PROVIDER_ERROR'
]

VALID_LABELS = ['none', 'slightly', 'moderately', 'highly']


def safe_parse_list(text):
    if pd.isna(text) or text in ERROR_TOKENS:
        return []
    try:
        parsed = ast.literal_eval(str(text))
        if isinstance(parsed, list):
            if len(parsed) == 1 and str(parsed[0]).lower() in ['no', 'none']:
                return []
            return parsed
        return []
    except Exception:
        return []


def span_pair_f1(a, b):
    a_tokens = set(str(a).split())
    b_tokens = set(str(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(a_tokens)
    recall = overlap / len(b_tokens)
    return 2 * precision * recall / (precision + recall)


def best_match_avg(source, target):
    if not source:
        return 0.0
    return sum(max(span_pair_f1(s, t) for t in target) for s in source) / len(source)


def calc_span_f1(gold, pred):
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    precision = best_match_avg(pred, gold)
    recall = best_match_avg(gold, pred)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_evaluation():
    print("loading bertscore model...\n")
    bertscore = evaluate.load("bertscore")

    results = []

    for config in FILES:
        file_path = os.path.join(BASE_DIR, config["file"])
        print(f"processing: {config['lang']} {config['model']} [{config['setup']}]")

        if not os.path.exists(file_path):
            print(f"  file missing: {config['file']}")
            continue

        df = pd.read_csv(file_path, dtype=str).fillna("")
        df = df[df['severity_raw'] != ""]

        if len(df) == 0:
            continue

        # drop rows with pipeline errors in any output column
        clean_mask = (
            (~df['severity_parsed'].isin(ERROR_TOKENS)) &
            (~df['spans_parsed'].isin(ERROR_TOKENS)) &
            (~df['rationales_parsed'].isin(ERROR_TOKENS)) &
            (~df['severity_parsed'].str.startswith(('FORMAT_ERROR', 'API_ERROR'), na=False)) &
            (~df['spans_parsed'].str.startswith(('FORMAT_ERROR', 'API_ERROR'), na=False)) &
            (~df['rationales_parsed'].str.startswith(('FORMAT_ERROR', 'API_ERROR'), na=False))
        )
        df_eval = df[clean_mask].copy()

        dropped = len(df) - len(df_eval)
        if dropped > 0:
            print(f"  filtered out {dropped}/{len(df)} rows from evaluation pool")

        if len(df_eval) == 0:
            continue

        # severity: 4-class macro F1
        gold_labels = df_eval['label_lower'].apply(lambda x: x if x in VALID_LABELS else 'none').tolist()
        pred_labels = df_eval['severity_parsed'].apply(lambda x: x if x in VALID_LABELS else 'none').tolist()
        severity_f1 = f1_score(gold_labels, pred_labels, average='macro', labels=VALID_LABELS, zero_division=0)

        # span detection: token overlap F1
        df_eval['gold_spans'] = df_eval['spans'].apply(safe_parse_list)
        df_eval['pred_spans'] = df_eval['spans_parsed'].apply(safe_parse_list)
        span_f1 = df_eval.apply(lambda r: calc_span_f1(r['gold_spans'], r['pred_spans']), axis=1).mean()

        # rationale quality: BERTScore F1 (on problematic items with gold rationales)
        df_eval['gold_rats'] = df_eval['rationales'].apply(safe_parse_list)
        df_eval['pred_rats'] = df_eval['rationales_parsed'].apply(safe_parse_list)
        df_rats = df_eval[df_eval['gold_rats'].apply(len) > 0].copy()

        if len(df_rats) == 0:
            bert_f1 = 0.0
        else:
            gold_texts = [" ".join(x) for x in df_rats['gold_rats']]
            pred_texts = [" ".join(x) for x in df_rats['pred_rats']]
            bs = bertscore.compute(predictions=pred_texts, references=gold_texts, lang=config["lang_code"])
            bert_f1 = float(np.mean(bs['f1'])) if bs['f1'] else 0.0

        results.append({
            "Language": config["lang"],
            "Model": config["model"],
            "Context": f"{config['lang']} - {config['model']}",
            "Setup": config["setup"],
            "Severity_Macro_F1": severity_f1,
            "Spans_Overlap_F1": span_f1,
            "Rationales_BERTScore": bert_f1
        })

    return pd.DataFrame(results)


def print_results(df):
    out = df.copy()
    for col in ['Severity_Macro_F1', 'Spans_Overlap_F1', 'Rationales_BERTScore']:
        out[col] = out[col].round(4)
    print("\n" + "="*95)
    print("COMPREHENSIVE PERFORMANCE SUMMARY MATRIX (ALL 4 CONFIGURATIONS)")
    print("="*95)
    print(out[['Context', 'Setup', 'Severity_Macro_F1', 'Spans_Overlap_F1', 'Rationales_BERTScore']].to_string(index=False))
    print("="*95 + "\n")


def plot_results(df):
    palette = {
        "B0 (Zero-Shot)":   "#DF9F9F",
        "B1 (Static)":      "#B0C4DE",
        "M1 (Dynamic)":     "#4682B4",
        "A1 (Target-Inst)": "#8FBC8F"
    }

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))
    fig.suptitle(
        'Full Architectural Tracking: B0 (Zero-Shot) vs. B1 (Static) vs. M1 (Dynamic) vs. A1 (Target-Inst)',
        fontsize=20, fontweight='bold', y=1.05
    )

    metrics = [
        ('Severity_Macro_F1',      'Task 1: Severity (Macro F1)',          axes[0]),
        ('Spans_Overlap_F1',       'Task 2a: Spans (Token Overlap F1)',     axes[1]),
        ('Rationales_BERTScore',   'Task 2b: Rationales (BERTScore F1)',    axes[2]),
    ]

    for col, title, ax in metrics:
        sns.barplot(data=df, x='Context', y=col, hue='Setup', ax=ax,
                    palette=palette, edgecolor='black', alpha=0.9)
        ax.set_title(title, fontweight='bold', pad=15)
        ax.set_ylabel('Score')
        ax.set_xlabel('')
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis='x', rotation=45)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'results_all_metrics.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"figure saved to {fig_path}")


if __name__ == '__main__':
    results_df = run_evaluation()
    print_results(results_df)
    plot_results(results_df)