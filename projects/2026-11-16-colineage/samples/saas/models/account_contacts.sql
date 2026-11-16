SELECT a.id AS account_id, a.region, u.email
FROM accounts AS a
LEFT JOIN users AS u ON u.account_id = a.id AND u.role = 'admin'
