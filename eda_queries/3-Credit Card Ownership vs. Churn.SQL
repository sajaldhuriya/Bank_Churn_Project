-- Q3: Are customers who hold a credit card more or less likely to churn?

SELECT 
    a.HasCreditCard,
    CASE WHEN a.HasCreditCard = 1 THEN 'Has Credit Card' ELSE 'No Credit Card' END AS CardStatus,
    COUNT(d.CustomerId) AS TotalCustomers,
    SUM(CAST(d.Churned AS INT)) AS ChurnedCustomers,
    ROUND(100.0 * SUM(CAST(d.Churned AS INT)) / COUNT(d.CustomerId), 2) AS ChurnRatePercent
FROM dbo.demographic d
JOIN dbo.account a ON d.CustomerId = a.CustomerId
GROUP BY a.HasCreditCard
ORDER BY ChurnRatePercent DESC;
