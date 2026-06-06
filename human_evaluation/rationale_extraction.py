# Human evaluation item selection
# This code extracts paired and asymmetric rationale items from Maverick B1 and M1 results for the blind A/B test
# Selection logic follows the criteria described in thesis Chapter 6, Section 6.6.
# Requirements: pip install pandas
# Set BASE_DIR to the folder containing your pipeline result CSVs before running.

import os
import ast
import random
import pandas as pd

BASE_DIR = './results'
OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)

FILES = {
    "Farsi": {
        "B1": os.path.join(BASE_DIR, 'fa_B1_maverick_results.csv'),
        "M1": os.path.join(BASE_DIR, 'fa_M1_maverick_results.csv')
    },
    "Italian": {
        "B1": os.path.join(BASE_DIR, 'it_B1_maverick_results.csv'),
        "M1": os.path.join(BASE_DIR, 'it_M1_maverick_results.csv')
    }
}


def safe_parse_list(text):
    if pd.isna(text) or text in [
        "FORMAT_ERROR", "API_ERROR", "SKIPPED_DUE_TO_SPANS_ERROR",
        "SHORT_CIRCUITED_DUE_TO_NONE_SEVERITY"
    ]:
        return []
    try:
        parsed = ast.literal_eval(str(text))
        if isinstance(parsed, list):
            if len(parsed) == 1 and str(parsed[0]).lower() in ["no", "none"]:
                return []
            return parsed
        return []
    except Exception:
        return []


def token_f1(span_a, span_b):
    tokens_a = set(str(span_a).split())
    tokens_b = set(str(span_b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    if overlap == 0:
        return 0.0
    precision = overlap / len(tokens_a)
    recall = overlap / len(tokens_b)
    return 2 * precision * recall / (precision + recall)


def select_rationale_pair(row):
    """
    Selects one rationale from B1 and one from M1 for a given item.
    Rule 1: pick the pair with >= 30% span overlap between B1 and M1 spans.
    Rule 2: if no cross-system overlap, pick rationales whose spans best align with the gold reference spans (>= 30% threshold).
    Rule 3: fall back to the first available rationale in each condition.
    """
    spans_b1   = row['spans_list_B1']
    spans_m1   = row['spans_list_M1']
    rats_b1    = row['rats_list_B1']
    rats_m1    = row['rats_list_M1']
    gold_spans = row['gold_spans_list']

    fallback_b1 = rats_b1[0] if rats_b1 else "No rationale generated"
    fallback_m1 = rats_m1[0] if rats_m1 else "No rationale generated"

    # rule 1: cross-system span overlap
    for b_idx, b_span in enumerate(spans_b1):
        for m_idx, m_span in enumerate(spans_m1):
            if token_f1(b_span, m_span) >= 0.30:
                if b_idx < len(rats_b1) and m_idx < len(rats_m1):
                    return rats_b1[b_idx], rats_m1[m_idx], "rule 1: cross-system overlap"

    # rule 2: gold reference alignment
    best_b1_rat, best_b1_score = fallback_b1, 0.0
    best_m1_rat, best_m1_score = fallback_m1, 0.0

    if gold_spans:
        for b_idx, b_span in enumerate(spans_b1):
            for g_span in gold_spans:
                score = token_f1(b_span, g_span)
                if score >= 0.30 and score > best_b1_score and b_idx < len(rats_b1):
                    best_b1_score = score
                    best_b1_rat = rats_b1[b_idx]

        for m_idx, m_span in enumerate(spans_m1):
            for g_span in gold_spans:
                score = token_f1(m_span, g_span)
                if score >= 0.30 and score > best_m1_score and m_idx < len(rats_m1):
                    best_m1_score = score
                    best_m1_rat = rats_m1[m_idx]

    if best_b1_rat != fallback_b1 or best_m1_rat != fallback_m1:
        return best_b1_rat, best_m1_rat, "rule 2: gold reference alignment"

    return fallback_b1, fallback_m1, "rule 3: positional fallback"


def build_eval_pool(lang, paths):
    print(f"\n{lang.upper()} — building evaluation pool")

    df_b1 = pd.read_csv(paths["B1"], dtype=str).fillna("")
    df_m1 = pd.read_csv(paths["M1"], dtype=str).fillna("")
    df = pd.merge(df_b1, df_m1, on='text_id', suffixes=('_B1', '_M1'))

    # drop rows with errors in severity or span columns
    clean_mask = (
        (~df['severity_parsed_B1'].str.contains('ERROR|API', case=False, na=False)) &
        (~df['severity_parsed_M1'].str.contains('ERROR|API', case=False, na=False)) &
        (~df['spans_parsed_B1'].str.contains('ERROR|API', case=False, na=False)) &
        (~df['spans_parsed_M1'].str.contains('ERROR|API', case=False, na=False))
    )
    df = df[clean_mask].copy()

    df['rats_list_B1']  = df['rationales_parsed_B1'].apply(safe_parse_list)
    df['rats_list_M1']  = df['rationales_parsed_M1'].apply(safe_parse_list)
    df['spans_list_B1'] = df['spans_parsed_B1'].apply(safe_parse_list)
    df['spans_list_M1'] = df['spans_parsed_M1'].apply(safe_parse_list)
    df['gold_spans_list'] = df['spans_B1'].apply(safe_parse_list)

    df['rats_count_B1'] = df['rats_list_B1'].apply(len)
    df['rats_count_M1'] = df['rats_list_M1'].apply(len)
    df['is_problematic'] = df['label_lower_B1'].str.strip().str.lower().isin(
        ['slightly', 'moderately', 'highly']
    )

    # stratum 1: both conditions produced rationales for a problematic item
    pool_s1 = df[
        (df['rats_count_B1'] > 0) &
        (df['rats_count_M1'] > 0) &
        (df['is_problematic'])
    ]

    # stratum 2: one condition produced a rationale, the other did not
    pool_s2 = pd.concat([
        df[(df['rats_count_B1'] == 0) & (df['rats_count_M1'] > 0) & (df['is_problematic'])],
        df[(df['rats_count_M1'] == 0) & (df['rats_count_B1'] > 0) & (df['is_problematic'])]
    ])

    sample_s1 = pool_s1.sample(n=min(10, len(pool_s1)), random_state=42)
    sample_s2 = pool_s2.sample(n=len(pool_s2), random_state=42)

    print(f"  stratum 1 eligible: {len(pool_s1)} | sampled: {len(sample_s1)}")
    print(f"  stratum 2 items: {len(sample_s2)}")

    rows = []

    # stratum 1: paired a/b evaluation
    for _, row in sample_s1.iterrows():
        rat_b1, rat_m1, rule = select_rationale_pair(row)

        # randomize which option is A and which is B
        if random.random() > 0.5:
            opt_a_setup, opt_a_rat = "B1", rat_b1
            opt_b_setup, opt_b_rat = "M1", rat_m1
        else:
            opt_a_setup, opt_a_rat = "M1", rat_m1
            opt_b_setup, opt_b_rat = "B1", rat_b1

        rows.append({
            "language": lang, "stratum": 1,
            "text_id": row['text_id'], "gold_label": row['label_lower_B1'],
            "news_text": row['text_B1'],
            "option_a_setup": opt_a_setup, "option_a_rationale": opt_a_rat,
            "option_b_setup": opt_b_setup, "option_b_rationale": opt_b_rat,
            "selection_rule": rule
        })

    # stratum 2: asymmetric detection items
    for _, row in sample_s2.iterrows():
        if row['rats_count_B1'] == 0:
            active_setup = "M1"
            active_rat = row['rats_list_M1'][0]
        else:
            active_setup = "B1"
            active_rat = row['rats_list_B1'][0]

        rows.append({
            "language": lang, "stratum": 2,
            "text_id": row['text_id'], "gold_label": row['label_lower_B1'],
            "news_text": row['text_B1'],
            "option_a_setup": active_setup, "option_a_rationale": active_rat,
            "option_b_setup": "N/A", "option_b_rationale": "N/A",
            "selection_rule": "asymmetric detection"
        })

    export_df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, f'{lang.lower()}_human_eval_mapping.csv')
    export_df.to_csv(out_path, index=False)
    print(f"  saved: {out_path}")


if __name__ == '__main__':
    for lang, paths in FILES.items():
        build_eval_pool(lang, paths)
    print("\ndone.")
