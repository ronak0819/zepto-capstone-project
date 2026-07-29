# Module 1 — Data Pipeline

This ETL pipeline scrapes the first five pages of [Books to Scrape](https://books.toscrape.com/), cleans at least 60 book records, converts prices using the assignment baseline of `1 GBP = 105.50 INR`, and loads normalized `categories` and `books` tables into SQLite.

```powershell
python data_pipeline\main.py
```

Invalid prices and ratings are median-imputed, ratings are constrained to 1–5, unusable identifying fields are dropped, duplicates are removed, and availability becomes Boolean. Median imputation is used because it is less sensitive to extreme values.

Generated files include raw and cleaned CSVs, `data/books.db`, six SQL-query CSVs, the SQL itself, and `outputs/join_comparison.csv`. The last output verifies that an SQL inner join and an equivalent `pandas.merge` return the same rows.
