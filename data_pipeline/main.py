from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "data"
OUTPUT_DIR = MODULE_DIR / "outputs"
DATABASE_PATH = DATA_DIR / "books.db"
BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR_RATE = 105.50
MINIMUM_BOOKS = 60
PAGES_TO_SCRAPE = 5
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_soup(url: str) -> BeautifulSoup:
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 ZeptoDataPipeline/1.0 (educational project)"},
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def get_book_category(book_url: str) -> str:
    category = fetch_soup(book_url).select_one("ul.breadcrumb > li:nth-of-type(3) > a")
    return category.get_text(strip=True) if category else "Unknown"


def scrape_books(page_count: int = PAGES_TO_SCRAPE) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for page_number in range(1, page_count + 1):
        page_url = urljoin(BASE_URL, f"catalogue/page-{page_number}.html")
        print(f"Scraping page {page_number}: {page_url}")
        for book in fetch_soup(page_url).select("article.product_pod"):
            title = book.select_one("h3 a")
            price = book.select_one("p.price_color")
            rating = book.select_one("p.star-rating")
            availability = book.select_one("p.instock.availability")
            if not all([title, price, rating, availability]):
                continue
            detail_url = urljoin(page_url, str(title.get("href", "")))
            records.append(
                {
                    "title": title.get("title", "").strip(),
                    "raw_price": price.get_text(strip=True),
                    "star_rating": next(
                        (value for value in rating.get("class", []) if value in RATING_MAP),
                        None,
                    ),
                    "availability": availability.get_text(" ", strip=True),
                    "category": get_book_category(detail_url),
                    "source_url": detail_url,
                }
            )
    return pd.DataFrame(records)


def clean_books(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("The scraper returned no book records.")
    cleaned = raw_df.copy()
    rows_before = len(cleaned)
    cleaned["price_gbp"] = pd.to_numeric(
        cleaned["raw_price"].str.replace(r"[^\d.]", "", regex=True), errors="coerce"
    )
    cleaned["rating"] = pd.to_numeric(cleaned["star_rating"].map(RATING_MAP), errors="coerce")
    cleaned["in_stock"] = (
        cleaned["availability"].fillna("").str.strip().str.lower().str.startswith("in stock")
    )
    missing_prices = int(cleaned["price_gbp"].isna().sum())
    missing_ratings = int(cleaned["rating"].isna().sum())
    if cleaned["price_gbp"].notna().any():
        cleaned["price_gbp"] = cleaned["price_gbp"].fillna(cleaned["price_gbp"].median())
    if cleaned["rating"].notna().any():
        cleaned["rating"] = cleaned["rating"].fillna(cleaned["rating"].median())
    cleaned["rating"] = cleaned["rating"].round().clip(1, 5).astype(int)
    cleaned = cleaned.dropna(subset=["title", "category", "price_gbp"])
    cleaned = cleaned[
        cleaned["title"].str.strip().ne("") & cleaned["category"].str.strip().ne("")
    ].copy()
    cleaned["price_gbp"] = cleaned["price_gbp"].astype(float)
    cleaned["price_inr"] = (cleaned["price_gbp"] * GBP_TO_INR_RATE).round(2)
    cleaned["in_stock"] = cleaned["in_stock"].astype(bool)
    cleaned = cleaned.drop_duplicates(["title", "category", "price_gbp"]).reset_index(drop=True)
    print(
        f"Rows before/after: {rows_before}/{len(cleaned)}; "
        f"imputed prices/ratings: {missing_prices}/{missing_ratings}"
    )
    if len(cleaned) < MINIMUM_BOOKS:
        raise ValueError(f"Only {len(cleaned)} records; at least {MINIMUM_BOOKS} are required.")
    if cleaned["category"].nunique() < 3:
        raise ValueError("The cleaned data contains fewer than three categories.")
    return cleaned[
        ["title", "price_gbp", "price_inr", "rating", "in_stock", "category", "source_url"]
    ]


def create_schema(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        DROP TABLE IF EXISTS books;
        DROP TABLE IF EXISTS categories;
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        );
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
            in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
            category_id INTEGER NOT NULL REFERENCES categories(category_id),
            source_url TEXT
        );
        """
    )


def load_database(connection: sqlite3.Connection, books_df: pd.DataFrame) -> None:
    connection.executemany(
        "INSERT INTO categories (category_name) VALUES (?)",
        [(category,) for category in sorted(books_df["category"].unique())],
    )
    lookup = pd.read_sql("SELECT category_id, category_name FROM categories", connection)
    database_books = books_df.merge(
        lookup, left_on="category", right_on="category_name", validate="many_to_one"
    )
    connection.executemany(
        """INSERT INTO books
        (title, price_gbp, price_inr, rating, in_stock, category_id, source_url)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                row.title,
                float(row.price_gbp),
                float(row.price_inr),
                int(row.rating),
                int(row.in_stock),
                int(row.category_id),
                row.source_url,
            )
            for row in database_books.itertuples(index=False)
        ],
    )
    connection.commit()


QUERIES = {
    "01_available_high_rated": """
        SELECT title, rating, price_gbp, price_inr FROM books
        WHERE in_stock = 1 AND rating >= 4 ORDER BY rating DESC, price_gbp DESC;
    """,
    "02_ten_most_expensive": """
        SELECT title, price_gbp, price_inr FROM books ORDER BY price_gbp DESC LIMIT 10;
    """,
    "03_distinct_categories": """
        SELECT DISTINCT category_name FROM categories ORDER BY category_name;
    """,
    "04_mid_priced_books": """
        SELECT title, price_gbp, rating FROM books
        WHERE price_gbp BETWEEN 20 AND 40 ORDER BY price_gbp;
    """,
    "05_four_or_five_star_books": """
        SELECT title, rating, price_gbp FROM books
        WHERE rating IN (4, 5) ORDER BY rating DESC, title;
    """,
    "06_books_with_categories": """
        SELECT b.book_id, b.title, b.price_gbp, b.price_inr, b.rating, b.in_stock,
               c.category_name
        FROM books b INNER JOIN categories c ON b.category_id = c.category_id
        ORDER BY c.category_name, b.rating DESC, b.title;
    """,
}


def run_queries(connection: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    OUTPUT_DIR.joinpath("queries.sql").write_text(
        "\n\n".join(f"-- {name}\n{query.strip()}" for name, query in QUERIES.items()),
        encoding="utf-8",
    )
    results = {}
    for name, query in QUERIES.items():
        results[name] = pd.read_sql(query, connection)
        results[name].to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        print(f"\n{name}\n{results[name].head(20).to_string(index=False)}")
    return results


def compare_sql_join_and_pandas_merge(
    connection: sqlite3.Connection, sql_join_df: pd.DataFrame
) -> None:
    books = pd.read_sql("SELECT * FROM books", connection)
    categories = pd.read_sql("SELECT * FROM categories", connection)
    pandas_join = (
        books.merge(categories, on="category_id", validate="many_to_one")[
            ["book_id", "title", "price_gbp", "price_inr", "rating", "in_stock", "category_name"]
        ]
        .sort_values(["category_name", "rating", "title"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    sql_join = sql_join_df.reset_index(drop=True)
    pd.testing.assert_frame_equal(sql_join, pandas_join, check_dtype=False)
    pd.concat({"sql_join": sql_join, "pandas_merge": pandas_join}, axis=1).to_csv(
        OUTPUT_DIR / "join_comparison.csv", index=False
    )
    print("\nSQL JOIN and pandas.merge outputs are equivalent.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ensure_directories()
    raw = scrape_books()
    raw.to_csv(DATA_DIR / "raw_books.csv", index=False)
    cleaned = clean_books(raw)
    cleaned.to_csv(DATA_DIR / "cleaned_books.csv", index=False)
    with sqlite3.connect(DATABASE_PATH) as connection:
        create_schema(connection)
        load_database(connection, cleaned)
        results = run_queries(connection)
        compare_sql_join_and_pandas_merge(connection, results["06_books_with_categories"])
    print(f"\nModule 1 completed. Database: {DATABASE_PATH}")


if __name__ == "__main__":
    main()
