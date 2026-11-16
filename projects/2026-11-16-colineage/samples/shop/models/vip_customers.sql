-- Above-average LTV and never refunded.
SELECT c.customer_id, c.name, c.ltv
FROM customer_ltv AS c
WHERE c.ltv > (SELECT avg(ltv) FROM customer_ltv)
  AND NOT EXISTS (
      SELECT 1 FROM refunds_enriched AS r WHERE r.customer_id = c.customer_id
  )
