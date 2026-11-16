-- Cleaned orders: test orders removed, status normalised, missing discount = 0.
SELECT
    id AS order_id,
    customer_id,
    lower(status) AS status,
    date(order_date) AS order_date,
    coalesce(discount, 0) AS discount
FROM orders
WHERE status <> 'test'
