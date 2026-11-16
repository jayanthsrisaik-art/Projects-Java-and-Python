SELECT
    c.id AS customer_id,
    c.name,
    upper(c.country) AS country,
    count(t.order_id) AS orders,
    coalesce(sum(t.net), 0) AS ltv,
    min(t.order_date) AS first_order
FROM customers AS c
LEFT JOIN order_totals AS t ON t.customer_id = c.id
GROUP BY c.id, c.name, c.country
