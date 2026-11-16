SELECT email AS contact, 'customer' AS kind FROM customers
UNION ALL
SELECT supplier AS contact, 'supplier' AS kind FROM products
