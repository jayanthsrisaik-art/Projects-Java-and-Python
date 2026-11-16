SELECT
    customer_id,
    ltv,
    rank() OVER (PARTITION BY country ORDER BY ltv DESC) AS country_rank
FROM customer_ltv
