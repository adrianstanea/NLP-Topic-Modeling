# NLP Topic Modeling

A Python package for topic modeling on Romanian text, specifically designed for the MOROCO (Moldavian and Romanian Dialectal Corpus) dataset.

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-green.svg)

## Overview

This project implements a TF-IDF + LDA hybrid pipeline for extracting coherent topics from Romanian news articles. It provides Romanian-specific preprocessing including diacritics normalization, dialect handling, and semantic POS filtering.

**Target Dataset:** MOROCO - Moldavian and Romanian Dialectal Corpus
**Categories:** Culture, Finance, Politics, Science, Sports, Tech (6 topics)

## Pipeline Architecture

![Pipeline Architecture](docs/figures/pipeline-architecture.png)

The pipeline consists of five main stages:

1. **Text Preprocessing** - Lowercase, diacritics normalization (ş→ș, ţ→ț), dialect normalization (î→â mid-word), NE token removal
2. **Tokenization** - spaCy-based tokenization with POS tagging, lemmatization, and semantic filtering (NOUN/PROPN/ADJ only)
3. **Stopword Filtering** - Combined NLTK + stopwordsiso + custom news boilerplate removal
4. **Hybrid Vectorization** - TF-IDF for feature selection, then BoW counts for LDA input
5. **LDA Topic Modeling** - Latent Dirichlet Allocation for topic extraction

## Project Structure

```
NLP-Topic-Modeling/
├── src/nlp_topic_modeling/          # Main package
│   ├── core/                        # Shared utilities (logging)
│   ├── data/                        # MOROCO dataset loader
│   ├── preprocessing/               # Text normalization, tokenization, stopwords
│   └── lda/                         # Topic modeling pipeline
├── tests/                           # Test suite
├── notebooks/                       # Jupyter notebooks for exploration
└── docs/figures/                    # Visualizations and diagrams
```

## Installation

### Prerequisites

- Python 3.12+
- uv package manager (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/adrianstanea/NLP-Topic-Modeling.git
cd NLP-Topic-Modeling

# Setup environment
uv venv
uv sync

# Download spaCy Romanian model
uv run python -m spacy download ro_core_news_sm
```

## Quick Start

```python
from nlp_topic_modeling.data.loaders import load_MOROCO
from nlp_topic_modeling.lda import train_topic_model

# Load MOROCO dataset
df, columns = load_MOROCO(split='train')

# Train topic model
pipeline = train_topic_model(df, n_topics=6)

# Display topics
pipeline.print_topics(n_words=10)
```

## Key Features

- **Romanian-specific preprocessing** - Handles cedilla-to-comma diacritics (ş→ș) and Moldovan dialect normalization (sînt→sânt)
- **Semantic POS filtering** - Keeps only NOUN, PROPN, ADJ for topical content, removing stylistic elements
- **Hybrid vectorization** - TF-IDF selects informative features, BoW counts preserve probabilistic integrity for LDA
- **Configurable pipeline** - Dataclass-based configuration for TF-IDF and LDA parameters

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test modules
uv run pytest tests/test_preprocessing.py -v
uv run pytest tests/test_lda.py -v

# With coverage
uv run pytest tests/ --cov=nlp_topic_modeling --cov-report=html
```

## Further Documentation

- See `notebooks/` for interactive examples and exploration
- See `docs/` for technical report and analysis documentation
