from datasets import load_dataset
import pandas as pd
from nlp_topic_modeling.core.logging import get_logger

logger = get_logger(__name__)

# Category mapping for MOROCO dataset (0-5)
MOROCO_CATEGORY_MAPPING = {
    0: "culture",
    1: "finance",
    2: "politics",
    3: "science",
    4: "sports",
    5: "tech",
}


def get_category_name(category_id: int) -> str:
    """
    Convert MOROCO category ID to category name.

    Args:
        category_id: Category ID (0-5)

    Returns:
        Category name
    """
    return MOROCO_CATEGORY_MAPPING.get(category_id, f"unknown_{category_id}")


def load_MOROCO(split: str = "train") -> tuple[pd.DataFrame, list]:
    """
    Load the MOROCO dataset from Hugging Face.

    Args:
        split: One of 'train', 'test', or 'validation'

    Returns:
        tuple: (DataFrame with 'sample' and 'category' columns, list of column names)

    Notes:
        - 'sample': Text content for processing
        - 'category': Numeric labels (0-5) for validation (use get_category_name() for readable names)
    """
    if split not in ["train", "test", "validation"]:
        raise ValueError(f"split must be one of 'train', 'test', or 'validation', got '{split}'")

    logger.info("Loading MOROCO dataset from Hugging Face")

    dataset = load_dataset(path="universityofbucharest/moroco", trust_remote_code=True)

    df = dataset[split].to_pandas()

    df.drop(columns=["id"], inplace=True, errors="ignore")

    logger.info(f"Loaded {len(df)} documents with {df['category'].nunique()} categories")
    logger.info(f"Category distribution: {df['category'].value_counts().sort_index().to_dict()}")

    columns = df.columns.tolist()
    return df, columns
