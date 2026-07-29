-- Q6: How does the number of bank products held by a customer influence their likelihood to churn?

SELECT 
    a.NumProducts,
    COUNT(d.CustomerId) AS TotalCustomers,
    SUM(CAST(d.Churned AS INT)) AS ChurnedCustomers,
    ROUND(100.0 * SUM(CAST(d.Churned AS INT)) / COUNT(d.CustomerId), 2) AS ChurnRatePercent
FROM dbo.demographic d
JOIN dbo.account a ON d.CustomerId = a.CustomerId
GROUP BY a.NumProducts
ORDER BY a.NumProducts ASC;
