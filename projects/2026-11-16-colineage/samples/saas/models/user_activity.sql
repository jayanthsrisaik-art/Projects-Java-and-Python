WITH ev AS (
    SELECT user_id, count(*) AS n_events, sum(duration) AS minutes, max(ts) AS last_seen
    FROM events
    WHERE event_type <> 'heartbeat'
    GROUP BY user_id
)
SELECT a.user_id, a.account_id, coalesce(ev.n_events, 0) AS n_events, ev.minutes, ev.last_seen
FROM active_users AS a
LEFT JOIN ev ON ev.user_id = a.user_id
