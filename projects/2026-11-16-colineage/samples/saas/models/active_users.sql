SELECT u.id AS user_id, u.account_id, lower(u.email) AS email
FROM users AS u
WHERE u.active = 1
