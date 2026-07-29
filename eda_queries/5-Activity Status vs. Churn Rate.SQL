-- Q5: How significantly does active member status impact customer retention?

SELECT 
    a.IsActive,
    CASE WHEN a.IsActive = 1 THEN 'Active Member' ELSE 'Inactive Member' END AS ActivityStatus,
    COUNT(d.CustomerId) AS TotalCustomers,
    SUM(CAST(d.Churned AS INT)) AS ChurnedCustomers,
    ROUND(100.0 * SUM(CAST(d.Churned AS INT)) / COUNT(d.CustomerId), 2) AS ChurnRatePercent
FROM dbo.demographic d
JOIN dbo.account a ON d.CustomerId = a.CustomerId
GROUP BY a.IsActive
ORDER BY ChurnRatePercent DESC;


