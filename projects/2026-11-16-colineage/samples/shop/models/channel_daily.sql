SELECT DISTINCT channel, date(order_date) AS day
FROM orders
WHERE status = 'complete'
