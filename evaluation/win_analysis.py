# Win analysis for rationale quality across all conditions, models, and languages.
# Computes BERTScore-based detection and quality advantage of M1 and A1 over B1.
# See thesis Chapter 6, Section 6.5 for full description of the analysis methodology.
#
# Requirements: pip install pandas numpy evaluate
# Set BASE_DIR to the folder containing your pipeline result CSVs before running.

import os
import ast
import numpy as np
import pandas as pd
import evaluate

BASE_DIR   = './results'
OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# items used for qualitative comparison in the thesis (farsi section)
FARSI_EXAMPLE_IDS = ['10427', '10105']


def safe_join(text):
    try:
        parsed = ast.literal_eval(str(text))
        if isinstance(parsed, list):
            if len(parsed) == 1 and str(parsed[0]).lower() in ['no', 'none']:
                return ''
            return ' '.join(parsed)
    except Exception:
        return ''
    return ''


def run_win_analysis(bertscore, lang_code, model_suffix, lang_label):
    print(f"\n{'='*65}")
    print(f"win analysis: {lang_label} — {model_suffix}")
    print(f"{'='*65}")

    b1 = pd.read_csv(os.path.join(BASE_DIR, f'{lang_code}_B1_{model_suffix}_results.csv'), dtype=str).fillna('')
    m1 = pd.read_csv(os.path.join(BASE_DIR, f'{lang_code}_M1_{model_suffix}_results.csv'), dtype=str).fillna('')
    a1 = pd.read_csv(os.path.join(BASE_DIR, f'{lang_code}_A1_{model_suffix}_results.csv'), dtype=str).fillna('')

    df = pd.merge(b1, m1, on='text_id', suffixes=('_B1', '_M1'))
    df = pd.merge(df, a1, on='text_id', suffixes=('', '_A1'))

    # keep only rows where all conditions produced clean output on problematic items
    clean = df[
        (~df['rationales_parsed_B1'].str.contains('ERROR|SKIP|CIRCUIT', na=False)) &
        (~df['rationales_parsed_M1'].str.contains('ERROR|SKIP|CIRCUIT', na=False)) &
        (~df['rationales_parsed'].str.contains('ERROR|SKIP|CIRCUIT', na=False)) &
        (df['label_lower_B1'].isin(['slightly', 'moderately', 'highly']))
    ].copy()

    clean['gold_text'] = clean['rationales_B1'].apply(safe_join)
    clean['b1_text']   = clean['rationales_parsed_B1'].apply(safe_join)
    clean['m1_text']   = clean['rationales_parsed_M1'].apply(safe_join)
    clean['a1_text']   = clean['rationales_parsed'].apply(safe_join)

    # compute BERTScore for each condition against gold reference
    for col, name in [('b1_text', 'bs_b1'), ('m1_text', 'bs_m1'), ('a1_text', 'bs_a1')]:
        result = bertscore.compute(
            predictions=clean[col].tolist(),
            references=clean['gold_text'].tolist(),
            lang=lang_code
        )
        clean[name] = result['f1']

    clean['m1_gain'] = clean['bs_m1'] - clean['bs_b1']
    clean['a1_gain'] = clean['bs_a1'] - clean['bs_b1']

    total = len(clean)

    b1_zero = (clean['bs_b1'] == 0).sum()
    m1_zero = (clean['bs_m1'] == 0).sum()
    a1_zero = (clean['bs_a1'] == 0).sum()

    fair_b1_m1 = clean[(clean['bs_b1'] > 0) & (clean['bs_m1'] > 0)]
    fair_b1_a1 = clean[(clean['bs_b1'] > 0) & (clean['bs_a1'] > 0)]
    fair_m1_a1 = clean[(clean['bs_m1'] > 0) & (clean['bs_a1'] > 0)]

    print(f"\nmissed detection ({total} problematic items)")
    print(f"  B1 misses: {b1_zero}/{total} ({100*b1_zero/total:.1f}%)")
    print(f"  M1 misses: {m1_zero}/{total} ({100*m1_zero/total:.1f}%)")
    print(f"  A1 misses: {a1_zero}/{total} ({100*a1_zero/total:.1f}%)")

    b1_miss_m1 = ((clean['bs_b1'] == 0) & (clean['bs_m1'] > 0)).sum()
    b1_miss_a1 = ((clean['bs_b1'] == 0) & (clean['bs_a1'] > 0)).sum()
    m1_miss_b1 = ((clean['bs_m1'] == 0) & (clean['bs_b1'] > 0)).sum()
    a1_miss_b1 = ((clean['bs_a1'] == 0) & (clean['bs_b1'] > 0)).sum()

    print(f"\ndetection advantage")
    print(f"  B1 missed, M1 caught: {b1_miss_m1}/{total} ({100*b1_miss_m1/total:.1f}%)")
    print(f"  B1 missed, A1 caught: {b1_miss_a1}/{total} ({100*b1_miss_a1/total:.1f}%)")
    print(f"  M1 missed, B1 caught: {m1_miss_b1}/{total} ({100*m1_miss_b1/total:.1f}%)")
    print(f"  A1 missed, B1 caught: {a1_miss_b1}/{total} ({100*a1_miss_b1/total:.1f}%)")

    m1_q_wins = (fair_b1_m1['m1_gain'] > 0).sum()
    b1_q_wins_m1 = (fair_b1_m1['m1_gain'] < 0).sum()
    a1_q_wins = (fair_b1_a1['a1_gain'] > 0).sum()
    b1_q_wins_a1 = (fair_b1_a1['a1_gain'] < 0).sum()
    m1_beats_a1 = (fair_m1_a1['m1_gain'] > fair_m1_a1['a1_gain']).sum()
    a1_beats_m1 = (fair_m1_a1['a1_gain'] > fair_m1_a1['m1_gain']).sum()
    tied = (fair_m1_a1['m1_gain'] == fair_m1_a1['a1_gain']).sum()

    n_bm = max(len(fair_b1_m1), 1)
    n_ba = max(len(fair_b1_a1), 1)
    n_ma = max(len(fair_m1_a1), 1)

    print(f"\nquality advantage (fair rows only — both conditions produced output)")
    print(f"  M1 beats B1: {m1_q_wins}/{len(fair_b1_m1)} ({100*m1_q_wins/n_bm:.1f}%)")
    print(f"  B1 beats M1: {b1_q_wins_m1}/{len(fair_b1_m1)} ({100*b1_q_wins_m1/n_bm:.1f}%)")
    print(f"  A1 beats B1: {a1_q_wins}/{len(fair_b1_a1)} ({100*a1_q_wins/n_ba:.1f}%)")
    print(f"  B1 beats A1: {b1_q_wins_a1}/{len(fair_b1_a1)} ({100*b1_q_wins_a1/n_ba:.1f}%)")
    print(f"  M1 beats A1: {m1_beats_a1}/{len(fair_m1_a1)} ({100*m1_beats_a1/n_ma:.1f}%)")
    print(f"  A1 beats M1: {a1_beats_m1}/{len(fair_m1_a1)} ({100*a1_beats_m1/n_ma:.1f}%)")
    print(f"  tied:        {tied}/{len(fair_m1_a1)} ({100*tied/n_ma:.1f}%)")

    m1_total = (clean['m1_gain'] > 0).sum()
    b1_total_m1 = (clean['m1_gain'] < 0).sum()
    a1_total = (clean['a1_gain'] > 0).sum()
    b1_total_a1 = (clean['a1_gain'] < 0).sum()

    print(f"\noverall advantage (detection + quality)")
    print(f"  M1 wins over B1: {m1_total}/{total} ({100*m1_total/total:.1f}%)")
    print(f"  B1 wins over M1: {b1_total_m1}/{total} ({100*b1_total_m1/total:.1f}%)")
    print(f"  A1 wins over B1: {a1_total}/{total} ({100*a1_total/total:.1f}%)")
    print(f"  B1 wins over A1: {b1_total_a1}/{total} ({100*b1_total_a1/total:.1f}%)")

    return clean


def print_example_items(clean, item_ids, lang_code):
    # prints rationale comparison for specific items used in thesis qualitative analysis
    print(f"\n{'='*65}")
    print(f"qualitative examples (thesis section 6.5)")
    print(f"{'='*65}")
    for tid in item_ids:
        rows = clean[clean['text_id'] == tid]
        if len(rows) == 0:
            print(f"\nitem {tid} not found in clean pool")
            continue
        row = rows.iloc[0]
        print(f"\nitem: {tid} | label: {row['label_lower_B1']}")
        print(f"bertscore — B1: {row['bs_b1']:.4f} | M1: {row['bs_m1']:.4f} | A1: {row['bs_a1']:.4f}")
        print(f"\nhuman reference:\n{row['gold_text']}")
        print(f"\nB1:\n{row['b1_text']}")
        print(f"\nM1:\n{row['m1_text']}")
        print(f"\nA1:\n{row['a1_text']}")
        print(f"\n{'-'*65}")


if __name__ == '__main__':
    print("loading bertscore model...")
    bertscore = evaluate.load("bertscore")

    fa_mav = run_win_analysis(bertscore, 'fa', 'maverick', 'Farsi Maverick')
    fa_mix = run_win_analysis(bertscore, 'fa', 'mixtral',  'Farsi Mixtral')
    it_mav = run_win_analysis(bertscore, 'it', 'maverick', 'Italian Maverick')
    it_mix = run_win_analysis(bertscore, 'it', 'mixtral',  'Italian Mixtral')

    # qualitative examples used in thesis chapter 6 farsi analysis
    print_example_items(fa_mix, FARSI_EXAMPLE_IDS, 'fa')

    print("\ndone.")
