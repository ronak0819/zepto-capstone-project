-- 01_available_high_rated
SELECT title, rating, price_gbp, price_inr FROM books
        WHERE in_stock = 1 AND rating >= 4 ORDER BY rating DESC, price_gbp DESC;

-- 02_ten_most_expensive
SELECT title, price_gbp, price_inr FROM books ORDER BY price_gbp DESC LIMIT 10;

-- 03_distinct_categories
SELECT DISTINCT category_name FROM categories ORDER BY category_name;

-- 04_mid_priced_books
SELECT title, price_gbp, rating FROM books
        WHERE price_gbp BETWEEN 20 AND 40 ORDER BY price_gbp;

-- 05_four_or_five_star_books
SELECT title, rating, price_gbp FROM books
        WHERE rating IN (4, 5) ORDER BY rating DESC, title;

-- 06_books_with_categories
SELECT b.book_id, b.title, b.price_gbp, b.price_inr, b.rating, b.in_stock,
               c.category_name
        FROM books b INNER JOIN categories c ON b.category_id = c.category_id
        ORDER BY c.category_name, b.rating DESC, b.title;