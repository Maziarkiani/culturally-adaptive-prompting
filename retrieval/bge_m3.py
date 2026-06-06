import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from google.colab import drive


# config:
# this script builds the bge-m3 retrieval banks used by the a1 pipelines
# each csv is encoded from the text column and saved as a pickle file
# change the paths to your specific input and output file paths
FA_INPUT_CSV = 'fa_exemplar_bank.csv'
FA_OUTPUT_PKL = 'fa_bge_m3_bank.pkl'

IT_INPUT_CSV = '/it_exemplar_bank.csv'
IT_OUTPUT_PKL = '/it_bge_m3_bank.pkl'


def mount_drive():
    drive.mount('/content/drive')


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading bge-m3 on device: {device}")
    model = SentenceTransformer('BAAI/bge-m3', device=device)
    return model


def build_index(input_csv, output_pkl, lang_name, model):
    df = pd.read_csv(input_csv).fillna("")
    print(f"loaded {len(df)} examples for {lang_name}")

    embeddings = model.encode(
        df['text'].tolist(),
        show_progress_bar=True,
        convert_to_tensor=False,
        normalize_embeddings=True
    )

    df['bge_m3_embedding'] = list(embeddings)

    output_dir = os.path.dirname(output_pkl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_pickle(output_pkl)
    print(f"saved {lang_name} index to {output_pkl}")


def main():
    mount_drive()
    model = load_model()

    build_index(FA_INPUT_CSV, FA_OUTPUT_PKL, "farsi", model)
    build_index(IT_INPUT_CSV, IT_OUTPUT_PKL, "italian", model)

    print("bge-m3 index build complete")


if __name__ == '__main__':
    main()