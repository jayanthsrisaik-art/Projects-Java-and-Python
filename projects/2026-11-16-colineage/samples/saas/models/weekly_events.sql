SELECT
    strftime('%Y-%W', e.ts) AS week,
    count(DISTINCT e.user_id) AS users,
    sum(sum(e.duration)) OVER (ORDER BY strftime('%Y-%W', e.ts)) AS running_minutes
FROM events AS e
JOIN active_users AS au ON au.user_id = e.user_id
GROUP BY 1
