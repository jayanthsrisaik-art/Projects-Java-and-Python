SELECT
    i.account_id,
    i.amount,
    i.amount / (SELECT sum(amount) FROM invoices WHERE currency = 'usd') AS share
FROM invoices AS i
WHERE i.currency = 'usd'
  AND i.account_id IN (SELECT account_id FROM active_users)
