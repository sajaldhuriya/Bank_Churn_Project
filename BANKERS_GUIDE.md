# 🏦 Banker's Operational Guide: Customer Retention System

This guide explains how to use the Churn Prediction tools to identify at-risk customers and implement retention strategies.

## 🎯 Goal
To transition from **reactive** churn management (asking why they left) to **proactive** retention (preventing them from leaving).

---

## 🛠️ The Toolset

### 1. The Strategic View (Power BI Dashboard)
**File:** `data_visualization/bank churn project.pbix`
*   **What it does:** Shows *trends* and *segments*.
*   **How to use it:** Use the dashboard to find "Churn Hotspots." 
    *   *Example:* "I notice that customers in Germany with only 1 product have a 40% churn rate."
    *   **Action:** This identifies a **systemic problem** that needs a marketing campaign or product change.

### 2. The Tactical View (ML Risk Model)
**The "Risk Score"**
*   **What it is:** The model assigns a probability between **0.0 (Safe)** and **1.0 (High Risk)** to every individual customer.
*   **How to interpret the score:**
    *   🔴 **Score > 0.7 (High Risk):** High probability of leaving. Requires immediate personalized intervention.
    *   🟡 **Score 0.4 - 0.7 (Medium Risk):** At risk. Include in "nudge" campaigns or wellness checks.
    *   🟢 **Score < 0.4 (Low Risk):** Stable customer. Maintain standard relationship.

---

## 📋 Action Playbook (The "So What?")

Based on the model's churn drivers, use these strategies to save customers:

| If the Driver is... | The Symptom is... | Recommended Action |
| :--- | :--- | :--- |
| **NumProducts** | Customer only has 1 product. | **Cross-sell Incentive:** Offer a discounted credit card or a high-interest savings account to "tie" them to the bank. |
| **BalancePerProduct** | Wealth is spread too thin or is very low. | **Wealth Management:** Offer a financial planning consultation or a tiered rewards account. |
| **IsActive** | Customer hasn't used the account in 3+ months. | **Re-engagement:** Send a "Welcome Back" offer or a feature update notification. |
| **AgeBucket** | Customer is in a high-risk age group (e.g., <30). | **Life-Stage Offer:** Offer student-friendly loans or first-time homebuyer workshops. |

---

## 🔄 Operational Workflow

To maximize retention, follow this monthly cycle:

1.  **Analyze Trends (Dashboard):** Identify which regions or product groups are struggling.
2.  **Generate Risk List (ML Model):** Run the model on your current customer base to get a list of individuals with a **Risk Score > 0.7**.
3.  **Match Strategy (Playbook):** Look at the driver for that specific customer (e.g., "only has 1 product").
4.  **Execute Outreach (CRM):** Contact the customer with the specific recommended offer.
5.  **Track Results:** Monitor if the churn rate for that segment drops in the next dashboard update.
