-- Q1: What is the total customer count, total churned customers, and the overall churn rate?

SELECT 
    COUNT(*) AS TotalCustomers,
    SUM(CAST(Churned AS INT)) AS TotalChurned,
    ROUND(100.0 * SUM(CAST(Churned AS INT)) / COUNT(*), 2) AS ChurnRatePercent
FROM dbo.demographic;