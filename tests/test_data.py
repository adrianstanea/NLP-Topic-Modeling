"""Tests for data loading functionality."""

import pytest
import pandas as pd
from nlp_topic_modeling.data.loaders import (
    load_MOROCO,
    get_category_name,
    MOROCO_CATEGORY_MAPPING
)


class TestCategoryMapping:
    """Tests for category name conversion."""

    def test_get_category_name_valid_ids(self):
        """Test that valid category IDs return correct names."""
        assert get_category_name(0) == 'culture'
        assert get_category_name(1) == 'finance'
        assert get_category_name(2) == 'politics'
        assert get_category_name(3) == 'science'
        assert get_category_name(4) == 'sports'
        assert get_category_name(5) == 'tech'

    def test_get_category_name_invalid_id(self):
        """Test that invalid category IDs return unknown format."""
        assert get_category_name(-1) == 'unknown_-1'
        assert get_category_name(6) == 'unknown_6'
        assert get_category_name(999) == 'unknown_999'

    def test_category_mapping_completeness(self):
        """Test that mapping contains all expected categories."""
        assert len(MOROCO_CATEGORY_MAPPING) == 6
        expected_categories = {'culture', 'finance', 'politics', 'science', 'sports', 'tech'}
        assert set(MOROCO_CATEGORY_MAPPING.values()) == expected_categories


class TestLoadMOROCO:
    """Tests for MOROCO dataset loading."""

    def test_load_moroco_train_split(self):
        """Test loading the train split."""
        df, columns = load_MOROCO(split='train')

        assert isinstance(df, pd.DataFrame)
        assert isinstance(columns, list)
        assert len(df) > 0
        assert 'sample' in df.columns
        assert 'category' in df.columns

    def test_load_moroco_test_split(self):
        """Test loading the test split."""
        df, columns = load_MOROCO(split='test')

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'sample' in df.columns
        assert 'category' in df.columns

    def test_load_moroco_validation_split(self):
        """Test loading the validation split."""
        df, columns = load_MOROCO(split='validation')

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'sample' in df.columns
        assert 'category' in df.columns

    def test_load_moroco_invalid_split(self):
        """Test that invalid split raises ValueError."""
        with pytest.raises(ValueError, match="split must be one of"):
            load_MOROCO(split='invalid')

    def test_category_values_numeric(self):
        """Test that category values remain numeric (0-5)."""
        df, _ = load_MOROCO(split='train')

        # Check that categories are numeric
        assert df['category'].dtype in [int, 'int64', 'int32']

        # Check that all values are in expected range
        assert df['category'].min() >= 0
        assert df['category'].max() <= 5

        # Check that all expected categories exist
        unique_categories = df['category'].unique()
        assert len(unique_categories) <= 6

    def test_required_columns_present(self):
        """Test that required columns are present."""
        df, _ = load_MOROCO(split='train')

        # Required columns should be present
        assert 'sample' in df.columns
        assert 'category' in df.columns

    def test_no_id_column(self):
        """Test that ID column is removed."""
        df, _ = load_MOROCO(split='train')

        assert 'id' not in df.columns

    def test_sample_column_text(self):
        """Test that sample column contains text data."""
        df, _ = load_MOROCO(split='train')

        # Check that sample contains strings
        assert df['sample'].dtype == object

        # Check that samples are non-empty
        assert (df['sample'].str.len() > 0).all()

    def test_columns_list_matches_dataframe(self):
        """Test that returned columns list matches DataFrame columns."""
        df, columns = load_MOROCO(split='train')

        assert columns == df.columns.tolist()


class TestIntegration:
    """Integration tests for data loading and category conversion."""

    def test_load_and_convert_categories(self):
        """Test loading data and converting category IDs to names."""
        df, _ = load_MOROCO(split='train')

        # Add readable category names
        df['category_name'] = df['category'].apply(get_category_name)

        # Check that conversion works
        assert 'category_name' in df.columns
        assert df['category_name'].isin(MOROCO_CATEGORY_MAPPING.values()).all()

        # Verify no unknown categories
        assert not df['category_name'].str.contains('unknown').any()
