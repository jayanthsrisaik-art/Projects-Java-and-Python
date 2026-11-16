-- One row per completed order with gross/net value.
WITH lines AS (
    SELECT order_id, sum(line_total) AS gross, count(*) AS n_lines
    FROM stg_items
    GROUP BY order_id
)
SELECT
    o.order_id,
    o.customer_id,
    o.order_date,
    l.gross,
    l.gross - o.discount AS net,
    l.n_lines
FROM stg_orders AS o
JOIN lines AS l ON l.order_id = o.order_id
WHERE o.status = 'complete'
