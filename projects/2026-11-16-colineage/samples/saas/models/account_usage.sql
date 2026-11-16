SELECT acc.id AS account_id, acc.plan, count(ua.user_id) AS users, sum(ua.minutes) AS minutes
FROM accounts AS acc
LEFT JOIN user_activity AS ua ON ua.account_id = acc.id
GROUP BY acc.id, acc.plan
