SELECT
    a.plan,
    sum(i.amount) AS billed,
    sum(CASE WHEN i.paid_at IS NOT NULL THEN i.amount ELSE 0 END) AS collected
FROM invoices AS i
JOIN accounts AS a ON a.id = i.account_id
WHERE i.currency = 'usd'
GROUP BY a.plan
HAVING count(*) >= 1
