SELECT region, plan, count(*) AS n FROM accounts GROUP BY region, plan
UNION
SELECT region, 'all' AS plan, count(*) AS n FROM accounts GROUP BY region
