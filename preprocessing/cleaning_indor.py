# InDor corpus cleaning pipeline
# 8 cleaning steps here. Read chapter 6, specifically, 6.1 of the thesis for more details on the cleaning process
# Set BASE_DIR and OUTPUT_DIR below before running.
import os
import pandas as pd
import numpy as np
from langdetect import detect, DetectorFactory
from sklearn.model_selection import train_test_split

DetectorFactory.seed = 42
RANDOM_SEED = 42
TEST_SET_SIZE = 100

BASE_DIR   = './data'
OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

path_it = os.path.join(BASE_DIR, 'indor_it.jsonl')
path_fa = os.path.join(BASE_DIR, 'indor_fa.jsonl')


# step 2a: manual blacklists
# article ids identified through human review as non-news content
# (advertisements, javascript error pages, non-target-language articles)
# see thesis chapter 6 for full description of the review process
# article id may be for unique but mutiple items in raw corpus based on number of annotations per item 

bad_ids_it = {
    6442, 6445, 6446, 6448, 6450, 6453, 6454, 6460, 6463, 6469, 6471, 6472,
    6475, 6477, 6480, 6494, 6495, 6505, 6514, 6515, 6553, 6572, 6577, 6584,
    6586, 6594, 6596, 6600, 6607, 6626, 6635, 6636, 6646, 6693, 6774, 6794,
    6840, 6841, 6850, 6888, 12292, 12329, 12488, 12492, 12499, 12549, 12556,
    12557, 12571, 12626, 12627, 12635, 12659, 12665, 12702, 12707, 12719,
    12737, 12746, 12747, 12750, 12765, 12792, 12824, 12834, 12835, 12846,
    12850, 12909, 12933, 12955, 12965, 12979, 13146, 14356, 14357, 14363,
    14364, 14366, 14382, 14397, 14434, 14435, 14471, 14479, 14484, 14586,
    14601, 14602, 14607, 14616, 14617, 14620, 14623, 14626, 14627, 14630,
    14666, 14667, 12303, 12388, 12760, 14473, 14599,
    6512, 6690, 6695, 12484, 12528, 12569, 12614, 12647, 12799, 12815,
    12866, 13113, 14344, 14536,
    6591, 12311, 12405, 12757, 12774, 12778, 14373, 14385, 14580, 14608, 14714
}

bad_ids_fa = {
    9567, 9572, 9577, 9578, 9580, 9583, 9586, 9588, 9592, 9593, 9595,
    9596, 9597, 9601, 9605, 9606, 9607, 9611, 9612, 9614, 9619, 9621,
    9623, 9630, 9633, 9690, 9692, 9723, 9746, 9749, 9777, 9789, 9795,
    9797, 9807, 9821, 9840, 9853, 9859, 9884, 9898, 9915, 9939, 9990,
    10002, 10032, 10051, 10074, 10078, 10079, 10080, 10100, 10101, 10103,
    10106, 10138, 10183, 10277, 10301, 10334, 10356, 10481, 10483, 10491,
    10524, 10530, 10538, 10539, 10558, 10005
}

# step 8: article-level integrity check (italian only)
# in the original run, these removals were identified through a targeted language detection scan and duplicate check on the clean pool after step 7.
# for reproducibility, the identified article ids are integrated here as part of the unified pipeline.
# Farsi was clean and no issues inside.
it_integrity_removals = [
    6437, 6470, 6492, 6496, 6497, 6500, 6517, 6548, 6549, 6550,
    6555, 6601, 6608, 6611, 6617, 6620, 6637, 6645, 6686, 6743,
    6775, 6779, 6780, 6803, 6839, 6842, 6843, 6847, 6892, 12310,
    12365, 12380, 12418, 12490, 12513, 12547, 12560, 12606, 12672,
    12847, 12855, 12883, 13142, 14349, 14418, 14652, 14719,
    12323, 12663, 12975, 6783, 12496, 6618, 6619
]


def is_empty_value(val):
    if isinstance(val, (list, np.ndarray)):
        return len(val) == 0
    if pd.isna(val):
        return True
    if str(val).strip() in ['[]', 'None', '', 'nan']:
        return True
    return False


def contains_any(series, patterns):
    if not patterns:
        return pd.Series(False, index=series.index)
    combined = '|'.join(f'(?:{p})' for p in patterns)
    return series.astype(str).str.contains(combined, regex=True, case=False, na=False)


def detect_lang_safe(text, fallback):
    try:
        clean = str(text).replace('[', '').replace(']', '').replace("'", "").strip()
        if len(clean) < 3:
            return fallback
        return detect(clean)
    except Exception:
        return fallback


def run_pipeline(file_path, lang_name, lang_code, bad_ids):
    print(f"\ncleaning pipeline: {lang_name}")

    df = pd.read_json(file_path, lines=True)
    print(f"raw annotations: {len(df)}")

    # step 1: remove invalid and missing labels
    n = len(df)
    df = df.dropna(subset=['label']).copy()
    df['label_lower'] = df['label'].astype(str).str.lower().str.strip()
    df = df[~df['label_lower'].isin(['n/a', 'nan', ''])].copy()
    print(f"step 1  remove invalid labels:               -{n - len(df)}")

    # step 2a: manual blacklist
    n = len(df)
    df = df[~df['text_id'].isin(bad_ids)].copy()
    print(f"step 2a manual blacklist:                    -{n - len(df)}")

    # step 2b: body-text regex filter for structural errors
    n = len(df)
    if lang_code == 'it':
        patterns = [
            r'devi attivare javascript',
            r'spiacent[ei]',
            r'attiva multiplayer\.it plus'
        ]
    elif lang_code == 'fa':
        patterns = [
            r'تولید\s+محتوای\s+بخش\s+[«"]?وب[‌ ]گردی[»"]?\s+توسط\s+این\s+مجموعه\s+صورت\s+نگرفته',
            r'فعال\s+بودن\s+جاوااسکریپت\s+الزامی\s+است'
        ]
    else:
        patterns = []
    df = df[~contains_any(df['text'], patterns)].copy()
    print(f"step 2b body-text regex filter:              -{n - len(df)}")

    # step 3: remove incomplete problematic annotations
    n = len(df)
    def has_missing_fields(row):
        if row['label_lower'] == 'none':
            return False
        return (is_empty_value(row['spans']) or
                is_empty_value(row['span_labels']) or
                is_empty_value(row['rationales']))
    df = df[~df.apply(has_missing_fields, axis=1)].copy()
    print(f"step 3  remove incomplete problematic items: -{n - len(df)}")

    # step 4: remove contaminated none annotations
    n = len(df)
    def is_contaminated_none(row):
        if row['label_lower'] == 'none':
            return (not is_empty_value(row['spans']) or
                    not is_empty_value(row['span_labels']) or
                    not is_empty_value(row['rationales']))
        return False
    df = df[~df.apply(is_contaminated_none, axis=1)].copy()
    print(f"step 4  remove contaminated none items:      -{n - len(df)}")

    # step 5: remove english rationale leakage
    n = len(df)
    prob_mask = df['label_lower'] != 'none'
    df.loc[prob_mask, 'detected_lang'] = (
        df.loc[prob_mask, 'rationales']
        .apply(lambda x: detect_lang_safe(x, lang_code))
    )
    df = df[~((df['label_lower'] != 'none') & (df['detected_lang'] == 'en'))].copy()
    print(f"step 5  remove english rationale leakage:    -{n - len(df)}")

    # step 6: exclude binary label conflicts
    n = len(df)
    df['is_problematic'] = df['label_lower'] != 'none'
    conflict_check = df.groupby('text_id')['is_problematic'].nunique()
    no_conflict_ids = conflict_check[conflict_check == 1].index
    df = df[df['text_id'].isin(no_conflict_ids)].copy()
    print(f"step 6  exclude binary conflicts:            -{n - len(df)}")

    # step 7: deduplicate keeping longest rationale
    n = len(df)
    df['rat_len'] = df['rationales'].astype(str).str.len()
    df = (df.sort_values(['text_id', 'rat_len'], ascending=[True, False])
            .drop_duplicates(subset=['text_id'], keep='first')
            .copy())
    print(f"step 7  deduplicate by rationale length:     -{n - len(df)}")

    # step 8: article-level integrity check (italian only)
    if lang_code == 'it':
        n = len(df)
        df = df[~df['text_id'].isin(it_integrity_removals)].copy()
        print(f"step 8  article-level integrity check:       -{n - len(df)}")
    else:
        print(f"step 8  article-level integrity check:       not applicable")

    df = df.drop(columns=[c for c in ['is_problematic', 'rat_len', 'detected_lang']
                           if c in df.columns])

    assert len(df) == df['text_id'].nunique(), "duplicate text_ids remain after cleaning"
    assert len(df) >= TEST_SET_SIZE, f"insufficient items after cleaning: {len(df)}"

    master_path = os.path.join(OUTPUT_DIR, f'{lang_code}_master_clean.csv')
    df.sort_values('text_id').to_csv(master_path, index=False, encoding='utf-8-sig')
    print(f"clean pool: {len(df)} articles saved")

    df_bank, df_test = train_test_split(
        df,
        test_size=TEST_SET_SIZE,
        random_state=RANDOM_SEED,
        stratify=df['label_lower']
    )
    df_test = df_test.sort_values('text_id').reset_index(drop=True)
    df_bank = df_bank.sort_values('text_id').reset_index(drop=True)

    assert len(set(df_test['text_id']) & set(df_bank['text_id'])) == 0, \
        "data leakage between test set and bank"

    df_test.to_csv(os.path.join(OUTPUT_DIR, f'{lang_code}_pilot_test.csv'),
                   index=False, encoding='utf-8-sig')
    df_bank.to_csv(os.path.join(OUTPUT_DIR, f'{lang_code}_exemplar_bank.csv'),
                   index=False, encoding='utf-8-sig')

    print(f"test set: {len(df_test)} | bank: {len(df_bank)}")


if __name__ == '__main__':
    run_pipeline(path_it, 'Italian', 'it', bad_ids_it)
    run_pipeline(path_fa, 'Farsi',   'fa', bad_ids_fa)
    print(f"\ndone. output saved to {OUTPUT_DIR}")