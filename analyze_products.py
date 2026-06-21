"""Analyze the products CSV dataset and print category-level statistics.

Core requirement:
    - number of products per category
    - average price and rating per category

Extras:
    - min/max price and total inventory value per category, in the same
      summary table as the core stats
    - top & lowest rated product per category
    - highest/lowest rated category, most/least expensive category
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tabulate import tabulate

NUMERIC_COLUMNS = ["price", "rating_rate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the products CSV dataset.")
    parser.add_argument(
        "--input", default="products_processed.csv", help="Path to the CSV produced by fetch_products.py"
    )
    return parser.parse_args()


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise SystemExit(f"Error: {csv_path} not found. Run fetch_products.py first.")

    df = pd.read_csv(csv_path)

    required_columns = {"id", "title", "category", "price", "rating_rate"}
    missing = required_columns - set(df.columns)
    if missing:
        raise SystemExit(f"Error: CSV is missing expected column(s): {sorted(missing)}")
    if df.empty:
        raise SystemExit(f"Error: {csv_path} contains no rows.")

    # Coerce price/rating to numeric and reject the file if anything doesn't
    # convert cleanly — a malformed or hand-edited CSV shouldn't be allowed
    # to silently produce NaN-laced (and therefore wrong) aggregates.
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_rows = df[NUMERIC_COLUMNS].isna().any(axis=1)
    if bad_rows.any():
        bad_ids = df.loc[bad_rows, "id"].tolist()
        preview = bad_ids[:10]
        suffix = "..." if len(bad_ids) > 10 else ""
        raise SystemExit(
            f"Error: non-numeric value(s) found in {NUMERIC_COLUMNS} for {bad_rows.sum()} "
            f"row(s) (ids: {preview}{suffix})"
        )

    return df


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-category overview: count, average price/rating, price range, inventory value."""
    return (
        df.groupby("category")
        .agg(
            product_count=("id", "count"),
            avg_price=("price", "mean"),
            avg_rating=("rating_rate", "mean"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            total_inventory_value=("price", "sum"),
        )
        .round(2)
        .sort_values(["product_count", "category"], ascending=[False, True])
        .reset_index()
    )


def top_and_bottom_rated(df: pd.DataFrame) -> pd.DataFrame:
    """Highest and lowest rated product within each category.

    Note: if a category has a tie for the top (or bottom) rating, idxmax/
    idxmin surface only one of the tied products - can be decided based on requirements.
    """
    idx_top = df.groupby("category")["rating_rate"].idxmax()
    idx_bottom = df.groupby("category")["rating_rate"].idxmin()

    top = df.loc[idx_top, ["category", "title", "rating_rate"]].rename(
        columns={"title": "top_rated_product", "rating_rate": "top_rating"}
    )
    bottom = df.loc[idx_bottom, ["category", "title", "rating_rate"]].rename(
        columns={"title": "lowest_rated_product", "rating_rate": "lowest_rating"}
    )
    return top.merge(bottom, on="category")


def category_extremes(df: pd.DataFrame) -> pd.DataFrame:
    """Which category rates highest/lowest, and which is most/least expensive, on average."""
    avg_rating = df.groupby("category")["rating_rate"].mean()
    avg_price = df.groupby("category")["price"].mean()

    rows = [
        ("Highest-rated category", avg_rating.idxmax(), round(avg_rating.max(), 2)),
        ("Lowest-rated category", avg_rating.idxmin(), round(avg_rating.min(), 2)),
        ("Most expensive category (avg price)", avg_price.idxmax(), round(avg_price.max(), 2)),
        ("Cheapest category (avg price)", avg_price.idxmin(), round(avg_price.min(), 2)),
    ]
    return pd.DataFrame(rows, columns=["metric", "category", "value"])


def print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))


def main() -> None:
    args = parse_args()
    df = load_dataset(Path(args.input))

    print(f"Loaded {len(df)} product(s) across {df['category'].nunique()} category(ies) from {args.input}")

    print_table("Category summary — count, avg price, avg rating, price range, inventory value", category_summary(df))
    print_table("Top & lowest rated product per category", top_and_bottom_rated(df))
    print_table("Category extremes", category_extremes(df))


if __name__ == "__main__":
    main()