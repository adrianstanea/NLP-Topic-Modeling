from datasets import load_dataset
import pandas as pd


def load_MOROCO(split: str = 'train') -> tuple[pd.DataFrame, list]:
    if split not in ['train', 'test', 'validation']:
        raise ValueError(f"split must be one of 'train', 'test', or 'validation', got '{split}'")

    print("--- Loading MOROCO from Hugging Face ---")

    dataset = load_dataset(
        path="universityofbucharest/moroco",
        trust_remote_code=True
    )

    df = dataset[split].to_pandas()
    df.drop(columns=['id'], inplace=True)
    text_col = 'sample' if 'sample' in df.columns else 'text'

    print(f"\tLoaded {len(df)} documents.")

    columns = df.columns.tolist()
    return df, columns