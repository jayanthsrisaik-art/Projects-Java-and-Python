SELECT
    p.category,
    sum(i.line_total) AS revenue,
    sum(i.line_total - i.quantity * p.cost) AS margin
FROM stg_items AS i
JOIN products AS p ON p.id = i.product_id
GROUP BY p.category
