-- Q8: Does customer loyalty (tenure in years) reduce customer churn?

SELECT 
    CASE 
        WHEN a.Tenure <= 2 THEN '0-2 Years'
        WHEN a.Tenure BETWEEN 3 AND 5 THEN '3-5 Years'
        WHEN a.Tenure BETWEEN 6 AND 8 THEN '6-8 Years'
        ELSE '9+ Years'
    END AS TenureGroup,
    COUNT(d.CustomerId) AS TotalCustomers,
    SUM(CAST(d.Churned AS INT)) AS ChurnedCustomers,
    ROUND(100.0 * SUM(CAST(d.Churned AS INT)) / COUNT(d.CustomerId), 2) AS ChurnRatePercent
FROM dbo.demographic d
JOIN dbo.account a ON d.CustomerId = a.CustomerId
GROUP BY 
    CASE 
        WHEN a.Tenure <= 2 THEN '0-2 Years'
        WHEN a.Tenure BETWEEN 3 AND 5 THEN '3-5 Years'
        WHEN a.Tenure BETWEEN 6 AND 8 THEN '6-8 Years'
        ELSE '9+ Years'
    END
ORDER BY ChurnRatePercent DESC;
