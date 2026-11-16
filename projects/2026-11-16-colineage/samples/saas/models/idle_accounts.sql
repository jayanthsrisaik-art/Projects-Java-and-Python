SELECT id AS account_id FROM accounts
EXCEPT
SELECT account_id FROM user_activity WHERE n_events > 0
