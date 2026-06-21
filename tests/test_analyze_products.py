"""Tests for analyze_products.py: load_dataset validation and every
aggregation function, against a small known dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import analyze_products as ap

SAMPLE_ROWS = [
    {"id": 1, "title": "Backpack", "price": 109.95, "description": "d", "category": "men's clothing", "image": "i", "rating_rate": 3.9, "rating_count": 120},
    {"id": 2, "title": "T-Shirt", "price": 22.3, "description": "d", "category": "men's clothing", "image": "i", "rating_rate": 4.1, "rating_count": 259},
    {"id": 3, "title": "Necklace", "price": 695.0, "description": "d", "category": "jewelery", "image": "i", "rating_rate": 4.6, "rating_count": 400},
    {"id": 4, "title": "Ring", "price": 9.99, "description": "d", "category": "jewelery", "image": "i", "rating_rate": 3.0, "rating_count": 70},
    {"id": 5, "title": "Laptop", "price": 999.99, "description": "d", "category": "electronics", "image": "i", "rating_rate": 4.8, "rating_count": 900},
]


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "products.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def sample_df(tmp_path):
    return ap.load_dataset(_write_csv(tmp_path, SAMPLE_ROWS))


# ---------------------------------------------------------------------------
# load_dataset: validation behavior
# ---------------------------------------------------------------------------

def test_load_dataset_missing_file_exits(tmp_path):
    with pytest.raises(SystemExit):
        ap.load_dataset(tmp_path / "nope.csv")


def test_load_dataset_missing_title_column_exits(tmp_path):
    # Regression test: title is required by top_and_bottom_rated() but was
    # once missing from required_columns, so this used to pass validation
    # and crash later instead of failing cleanly here.
    rows = [{k: v for k, v in row.items() if k != "title"} for row in SAMPLE_ROWS]
    path = _write_csv(tmp_path, rows)

    with pytest.raises(SystemExit):
        ap.load_dataset(path)


def test_load_dataset_missing_category_column_exits(tmp_path):
    rows = [{k: v for k, v in row.items() if k != "category"} for row in SAMPLE_ROWS]
    path = _write_csv(tmp_path, rows)

    with pytest.raises(SystemExit):
        ap.load_dataset(path)


def test_load_dataset_empty_file_exits(tmp_path):
    path = tmp_path / "products.csv"
    pd.DataFrame(columns=list(SAMPLE_ROWS[0])).to_csv(path, index=False)

    with pytest.raises(SystemExit):
        ap.load_dataset(path)


def test_load_dataset_non_numeric_price_exits(tmp_path):
    rows = [dict(row) for row in SAMPLE_ROWS]
    rows[1]["price"] = "not-a-number"
    path = _write_csv(tmp_path, rows)

    with pytest.raises(SystemExit):
        ap.load_dataset(path)


def test_load_dataset_non_numeric_rating_exits(tmp_path):
    rows = [dict(row) for row in SAMPLE_ROWS]
    rows[3]["rating_rate"] = "five stars"
    path = _write_csv(tmp_path, rows)

    with pytest.raises(SystemExit):
        ap.load_dataset(path)


def test_load_dataset_valid_file_loads_correctly(tmp_path):
    df = ap.load_dataset(_write_csv(tmp_path, SAMPLE_ROWS))

    assert len(df) == 5
    assert set(df["category"]) == {"men's clothing", "jewelery", "electronics"}


# ---------------------------------------------------------------------------
# category_summary
# ---------------------------------------------------------------------------

def test_category_summary_counts_and_averages(sample_df):
    by_category = ap.category_summary(sample_df).set_index("category")

    assert by_category.loc["jewelery", "product_count"] == 2
    assert by_category.loc["jewelery", "avg_price"] == pytest.approx((695.0 + 9.99) / 2, rel=1e-3)
    assert by_category.loc["jewelery", "avg_rating"] == pytest.approx((4.6 + 3.0) / 2, rel=1e-3)
    assert by_category.loc["jewelery", "min_price"] == 9.99
    assert by_category.loc["jewelery", "max_price"] == 695.0
    assert by_category.loc["jewelery", "total_inventory_value"] == pytest.approx(704.99, rel=1e-3)

    assert by_category.loc["electronics", "product_count"] == 1


def test_category_summary_sorts_by_count_desc_then_category_asc(sample_df):
    summary = ap.category_summary(sample_df)

    # men's clothing and jewelery are tied at count=2 -- tied categories must
    # be alphabetically ordered, not arbitrary.
    tied = summary[summary["product_count"] == 2]["category"].tolist()
    assert tied == sorted(tied)
    # electronics (count=1) must sort after both, since count desc is primary.
    assert summary.iloc[-1]["category"] == "electronics"


# ---------------------------------------------------------------------------
# top_and_bottom_rated
# ---------------------------------------------------------------------------

def test_top_and_bottom_rated_per_category(sample_df):
    result = ap.top_and_bottom_rated(sample_df).set_index("category")

    assert result.loc["men's clothing", "top_rated_product"] == "T-Shirt"
    assert result.loc["men's clothing", "lowest_rated_product"] == "Backpack"
    assert result.loc["jewelery", "top_rated_product"] == "Necklace"
    assert result.loc["jewelery", "lowest_rated_product"] == "Ring"
    # A single-product category is, trivially, both its own top and bottom.
    assert result.loc["electronics", "top_rated_product"] == "Laptop"
    assert result.loc["electronics", "lowest_rated_product"] == "Laptop"


# ---------------------------------------------------------------------------
# category_extremes
# ---------------------------------------------------------------------------

def test_category_extremes(sample_df):
    extremes = ap.category_extremes(sample_df).set_index("metric")

    # avg ratings: men's clothing=4.0, jewelery=3.8, electronics=4.8
    assert extremes.loc["Highest-rated category", "category"] == "electronics"
    assert extremes.loc["Lowest-rated category", "category"] == "jewelery"
    # avg prices: men's clothing=66.125, jewelery=352.495, electronics=999.99
    assert extremes.loc["Most expensive category (avg price)", "category"] == "electronics"
    assert extremes.loc["Cheapest category (avg price)", "category"] == "men's clothing"