-- SELECT * expands to refunds' columns; downstream lineage follows the expansion.
SELECT r.*, o.customer_id
FROM refunds AS r
JOIN stg_orders AS o ON o.order_id = r.order_id
