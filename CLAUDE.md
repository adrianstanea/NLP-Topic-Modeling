# CLAUDE.md - NLP Topic Modeling Project Guide

This document provides comprehensive guidance for working with and extending the NLP Topic Modeling project.

## Project Overview

A Python package for topic modeling on Romanian text, specifically designed for the MOROCO (Moldavian and Romanian Dialectal Corpus) dataset. The package provides Romanian-specific preprocessing and a TF-IDF + LDA hybrid pipeline for topic extraction.

**Primary Use Case**: Extract coherent topics from Romanian news articles across 6 categories (culture, finance, politics, science, sports, tech).

## Repository Structure

```
NLP-Topic-Modeling/
├── src/nlp_topic_modeling/          # Main package
│   ├── __init__.py
│   ├── core/                        # Shared utilities
│   │   ├── __init__.py
│   │   └── logging.py               # Centralized logging
│   ├── data/                        # Data loading
│   │   ├── __init__.py
│   │   └── loaders.py               # MOROCO dataset loader
│   ├── preprocessing/               # Text preprocessing
│   │   ├── __init__.py              # Public API exports
│   │   ├── normalizers.py           # Text normalization functions
│   │   ├── stopwords.py             # Stopword management
│   │   ├── tokenizer.py             # spaCy-based tokenization
│   │   └── pipeline.py              # RomanianPreprocessor class
│   └── lda/                         # Topic modeling
│       ├── __init__.py              # Public API + train_topic_model()
│       ├── config.py                # Dataclass configurations
│       ├── vectorizer.py            # HybridVectorizer (TF-IDF + BoW)
│       ├── model.py                 # TopicModel (LDA wrapper)
│       └── pipeline.py              # TopicModelingPipeline
├── tests/                           # Test suite
│   ├── test_preprocessing.py        # Preprocessing tests
│   ├── test_lda.py                  # LDA pipeline tests
│   └── test_data.py                 # Data loader tests
├── notebooks/                       # Jupyter notebooks
│   ├── 01-data-exploration.ipynb    # Dataset exploration
│   └── 02-feature-demo.ipynb        # Full feature demonstration
├── pyproject.toml                   # Package configuration
└── uv.lock                          # Dependency lock file
```

## Architecture

### Design Principles

1. **Filter-then-Feed**: TF-IDF selects top-K informative features, then BoW counts (not TF-IDF weights) feed LDA to preserve probabilistic integrity.

2. **Semantic Compression**: POS filtering keeps only NOUN, PROPN, ADJ (the "topical skeleton"), removing verbs (writing style).

3. **Cross-Dialect Consistency**: Normalizes RO-MD orthographic variation (î→â mid-word) for unified vocabulary.

4. **Modular Pipeline**: Each component (normalization, tokenization, vectorization, modeling) is independently configurable.

### Data Flow

```
Raw Text (MOROCO)
    ↓
Preprocessing
    ├── lowercase()
    ├── normalize_diacritics()      # ş→ș, ţ→ț
    ├── normalize_dialect()         # sînt→sânt (mid-word î→â)
    ├── remove_ne_tokens()          # $NE$ placeholder removal
    └── clean_whitespace()
    ↓
Tokenization (spaCy ro_core_news_sm)
    ├── POS tagging
    ├── Lemmatization
    └── POS filtering (NOUN/PROPN/ADJ only)
    ↓
Stopword Filtering
    ├── NLTK + stop-words + stopwordsiso
    └── News boilerplate (verbal noise, web artifacts)
    ↓
Vectorization (HybridVectorizer)
    ├── TF-IDF for feature selection (max_df=0.4, min_df=5)
    └── BoW counts for selected features → LDA
    ↓
LDA Topic Modeling
    ├── sklearn LatentDirichletAllocation
    ├── n_topics=6 (matches MOROCO categories)
    └── online learning for large datasets
    ↓
Output: Topic distributions, top words per topic
```

## Key Components

### 1. Data Loading (`data/loaders.py`)

```python
from nlp_topic_modeling.data.loaders import load_MOROCO, get_category_name

df, columns = load_MOROCO(split='train')  # or 'test', 'validation'
```

- Loads MOROCO from Hugging Face
- Returns DataFrame with `sample` (text) and `category` (0-5) columns
- Use `get_category_name(id)` for readable category names

### 2. Preprocessing (`preprocessing/`)

```python
from nlp_topic_modeling.preprocessing import (
    normalize_diacritics,    # ş→ș, ţ→ț (cedilla→comma-below)
    normalize_dialect,       # sînt→sânt (mid-word î→â)
    remove_ne_tokens,        # Remove $NE$ placeholders
    RomanianTokenizer,       # spaCy-based tokenizer
    get_all_stopwords,       # Combined stopword list
    TOPIC_POS_TAGS,          # {'NOUN', 'PROPN', 'ADJ'}
)

tokenizer = RomanianTokenizer()
tokens = tokenizer.tokenize_pos_filtered(text)  # POS-filtered lemmas
```

**Key Normalizers:**
- `normalize_diacritics()`: Converts legacy cedilla (ş,ţ) to Unicode comma-below (ș,ț)
- `normalize_dialect()`: Converts Moldovan î to Romanian â (mid-word only)
- `remove_ne_tokens()`: Removes MOROCO's `$NE$` named entity placeholders

**Stopwords:** Combines NLTK, stop-words, stopwordsiso + news boilerplate (verbal noise like "declara", web artifacts like "foto", "video")

### 3. LDA Pipeline (`lda/`)

```python
from nlp_topic_modeling.lda import train_topic_model, TopicModelingPipeline

# Quick start
pipeline = train_topic_model(df, n_topics=6)
pipeline.print_topics()

# Or with custom config
from nlp_topic_modeling.lda import PipelineConfig, TFIDFConfig, LDAConfig

config = PipelineConfig(
    tfidf=TFIDFConfig(max_df=0.5, min_df=3, max_features=3000),
    lda=LDAConfig(n_topics=8, max_iter=500),
)
pipeline = TopicModelingPipeline(config)
doc_topics = pipeline.fit_transform(df)
```

**Configuration Defaults:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_df` | 0.4 | Prune terms in >40% of docs |
| `min_df` | 5 | Prune terms in <5 docs |
| `max_features` | 5000 | Feature density principle |
| `ngram_range` | (1, 2) | Unigrams + bigrams |
| `sublinear_tf` | True | 1 + log(tf) for length normalization |
| `n_topics` | 6 | Matches MOROCO categories |
| `learning_method` | 'online' | Efficient for large datasets |

## Development Setup

### Prerequisites

- Python 3.12+
- uv package manager (recommended) or pip

### Installation

```bash
# Clone repository
git clone https://github.com/adrianstanea/NLP-Topic-Modeling.git
cd NLP-Topic-Modeling

# Create virtual environment and install
uv venv
source .venv/bin/activate  # Linux/Mac
uv pip install -e ".[dev,ro]"

# Download spaCy Romanian model
python -m spacy download ro_core_news_sm
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific modules
pytest tests/test_preprocessing.py -v
pytest tests/test_lda.py -v

# With coverage
pytest tests/ --cov=nlp_topic_modeling --cov-report=html
```

## Coding Conventions

### 1. Type Hints
All functions use Python 3.12+ type hints:
```python
def normalize_dialect(text: str) -> str:
    ...

def train_topic_model(
    df: pd.DataFrame,
    n_topics: int = 6,
    **config_kwargs
) -> TopicModelingPipeline:
    ...
```

### 2. Dataclasses for Configuration
Use `@dataclass` for configuration objects:
```python
@dataclass
class TFIDFConfig:
    max_df: float = 0.4
    min_df: int = 5
    # ... with docstrings for each field
```

### 3. Logging
Use the centralized logger:
```python
from nlp_topic_modeling.core.logging import get_logger

logger = get_logger(__name__)
logger.info("Processing complete")
```

### 4. Public API via `__init__.py`
Export public functions/classes in `__init__.py`:
```python
# In preprocessing/__init__.py
from .normalizers import normalize_diacritics, normalize_dialect
# ...
__all__ = ['normalize_diacritics', 'normalize_dialect', ...]
```

### 5. Test Organization
- One test file per module: `test_preprocessing.py`, `test_lda.py`
- Test class per component: `TestDialectNormalization`, `TestHybridVectorizer`
- Use `pytest.fixture` for shared test data
- Use `@pytest.mark.skipif` for tests requiring optional dependencies

## Extending the Project

### Adding a New Preprocessing Step

1. Add function to `preprocessing/normalizers.py`:
```python
def remove_numbers(text: str) -> str:
    """Remove numeric characters from text."""
    return re.sub(r'\d+', '', text)
```

2. Export in `preprocessing/__init__.py`:
```python
from .normalizers import remove_numbers
__all__ = [..., 'remove_numbers']
```

3. Add tests in `tests/test_preprocessing.py`:
```python
class TestRemoveNumbers:
    def test_removes_digits(self):
        assert remove_numbers("abc123def") == "abcdef"
```

### Adding a New Topic Model (e.g., BERTopic)

1. Create module structure:
```
src/nlp_topic_modeling/
└── bertopic/
    ├── __init__.py
    ├── config.py
    ├── model.py
    └── pipeline.py
```

2. Follow the LDA pattern:
   - `config.py`: Dataclass configurations
   - `model.py`: Model wrapper with analysis utilities
   - `pipeline.py`: Full pipeline orchestration
   - `__init__.py`: Public API with convenience function

3. Reuse preprocessing:
```python
from nlp_topic_modeling.preprocessing import RomanianTokenizer, normalize_dialect
```

### Adding New Stopwords

Edit `preprocessing/stopwords.py`:
```python
def get_custom_stopwords() -> set[str]:
    """Custom domain-specific stopwords."""
    return {'word1', 'word2', ...}

def get_all_stopwords(...):
    all_stopwords.update(get_custom_stopwords())  # Add to combined set
```

### Adding a New Data Loader

1. Add to `data/loaders.py`:
```python
def load_custom_dataset(path: str) -> tuple[pd.DataFrame, list]:
    """Load custom Romanian text dataset."""
    logger.info(f"Loading dataset from {path}")
    # ... loading logic
    return df, columns
```

2. Export in `data/__init__.py`

## Requirements Reference

The project implements these design requirements:

| ID | Requirement | Implementation |
|----|-------------|----------------|
| R3.1 | Frequency pruning | `max_df=0.4`, `min_df=5` in TFIDFConfig |
| R3.2 | Bigram support | `ngram_range=(1,2)` in TFIDFConfig |
| R3.3 | Dialect normalization | `normalize_dialect()` (î→â mid-word) |
| R4.1 | POS filtering | `TOPIC_POS_TAGS = {'NOUN', 'PROPN', 'ADJ'}` |
| R4.2 | News boilerplate | `get_news_boilerplate_stopwords()` |

## Common Tasks

### Train a Topic Model
```python
from nlp_topic_modeling.data.loaders import load_MOROCO
from nlp_topic_modeling.lda import train_topic_model

df, _ = load_MOROCO()
pipeline = train_topic_model(df, n_topics=6)
pipeline.print_topics(n_words=10)
```

### Preprocess Text
```python
from nlp_topic_modeling.preprocessing import preprocess_text

result = preprocess_text("Eu sînt în România și văd ştiri.")
print(result.clean_text)  # Normalized, lowercased, stopwords removed
print(result.lemmas)       # Lemmatized tokens
```

### Get Document-Topic Distributions
```python
doc_topics = pipeline.fit_transform(df)  # Shape: (n_docs, n_topics)
dominant = pipeline.get_dominant_topics(doc_topics)  # Most likely topic per doc
```

### Custom POS Filtering
```python
from nlp_topic_modeling.preprocessing import RomanianTokenizer

tokenizer = RomanianTokenizer()
# Only nouns
nouns = tokenizer.tokenize_pos_filtered(text, include_pos={'NOUN'})
# Nouns + verbs
tokens = tokenizer.tokenize_pos_filtered(text, include_pos={'NOUN', 'VERB'})
```

## Troubleshooting

### spaCy Model Not Found
```bash
python -m spacy download ro_core_news_sm
```

### NLTK Stopwords Missing
```python
import nltk
nltk.download('stopwords')
```

### Memory Issues with Large Dataset
- Use sampling: `df.sample(10000, random_state=42)`
- Reduce `max_features` in TFIDFConfig
- Use `learning_method='online'` with smaller `batch_size`

## Dependencies

**Core:**
- pandas, numpy, scipy
- scikit-learn (TF-IDF, LDA)
- spacy (tokenization, POS tagging)
- datasets (Hugging Face MOROCO loader)

**Stopwords:**
- nltk, stop-words, stopwordsiso

**Visualization:**
- matplotlib, seaborn, pyldavis

**Future (BERTopic):**
- bertopic, transformers, torch, sentencepiece
