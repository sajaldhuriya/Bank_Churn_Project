-- Q4: Which regions or geographies experience the highest customer churn rate?

SELECT 
    l.Geography,
    COUNT(d.CustomerId) AS TotalCustomers,
    SUM(CAST(d.Churned AS INT)) AS ChurnedCustomers,
    ROUND(100.0 * SUM(CAST(d.Churned AS INT)) / COUNT(d.CustomerId), 2) AS ChurnRatePercent
FROM dbo.demographic d
JOIN dbo.location l ON d.LocationId = l.LocationId
GROUP BY l.Geography
ORDER BY ChurnRatePercent DESC;


