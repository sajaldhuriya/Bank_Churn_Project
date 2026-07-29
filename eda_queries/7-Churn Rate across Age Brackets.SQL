-- Q7: Which age groups exhibit the highest risk of churn?

SELECT 
    CASE 
        WHEN Age < 30 THEN '< 30'
        WHEN Age BETWEEN 30 AND 45 THEN '30 - 45'
        WHEN Age BETWEEN 46 AND 60 THEN '46 - 60'
        ELSE '> 60'
    END AS AgeGroup,
    COUNT(*) AS TotalCustomers,
    SUM(CAST(Churned AS INT)) AS ChurnedCustomers,
    ROUND(100.0 * SUM(CAST(Churned AS INT)) / COUNT(*), 2) AS ChurnRatePercent
FROM dbo.demographic
GROUP BY 
    CASE 
        WHEN Age < 30 THEN '< 30'
        WHEN Age BETWEEN 30 AND 45 THEN '30 - 45'
        WHEN Age BETWEEN 46 AND 60 THEN '46 - 60'
        ELSE '> 60'
    END
ORDER BY ChurnRatePercent DESC;