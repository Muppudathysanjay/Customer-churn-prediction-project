# Customer Churn — Business Report

**Project:** Customer Churn Prediction
**Dataset:** 7,043 customers (post-cleaning)
**Overall churn rate:** 34.2%

---

## 1. Executive Summary

Roughly **1 in 3 customers** churns. The strongest, most actionable drivers of churn are
**contract type**, **tenure**, **customer support call volume**, and **payment method** —
all of which the business can influence directly through retention offers, onboarding
programs, and support-process changes. A Logistic Regression model trained on this data
achieves a **ROC-AUC of 0.79** and a **74% recall** on the churn class, meaning it correctly
flags roughly 3 out of every 4 customers who are about to churn — enough signal to power a
proactive, prioritized retention program.

## 2. Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression (Selected)** | 0.710 | 0.559 | **0.737** | 0.636 | **0.790** |
| Random Forest | 0.711 | 0.567 | 0.665 | 0.612 | 0.775 |
| Support Vector Machine | 0.708 | 0.558 | 0.712 | 0.626 | 0.771 |
| XGBoost | 0.697 | 0.549 | 0.652 | 0.596 | 0.763 |
| Decision Tree | 0.669 | 0.513 | 0.710 | 0.596 | 0.745 |

**Why Logistic Regression was selected:** it achieved the highest ROC-AUC and the highest
recall on the churn class. In a churn use case, **recall matters more than precision** —
the cost of missing an at-risk customer (lost revenue, lost lifetime value) is typically
far higher than the cost of a retention offer extended to a customer who wasn't going to
churn anyway. Logistic Regression is also fully interpretable: every coefficient has a
direct, explainable business meaning, which matters for stakeholder trust and regulatory
comfort in customer-facing decisions.

## 3. Key Churn Drivers (Why Customers Leave)

1. **Contract type.** Month-to-month customers churn at **43.2%**, vs. 29.4% for one-year
   and **17.5%** for two-year contracts. Contract commitment is the single largest lever
   the business controls.
2. **Tenure.** Churned customers average **24.5 months** of tenure vs. **38.1 months** for
   retained customers. Risk is heavily front-loaded in the first year.
3. **Customer support calls.** Churn rate rises sharply once a customer places 4 or more
   support calls — a clear, early, actionable warning signal.
4. **Monthly charges.** Churned customers pay **$68.23/month** on average vs. **$58.59** for
   retained customers, pointing to a value-for-money perception gap rather than blanket
   price sensitivity.
5. **Payment method.** Electronic check payers churn more than customers on automatic
   payment methods (credit card, bank transfer) — a "friction" payment method correlates
   with lower commitment overall.
6. **Service attachment.** Customers without Online Security or Tech Support add-ons churn
   more easily; these services increase switching cost and perceived value.

## 4. High-Risk Customer Profile

A customer carries elevated churn risk when they exhibit **most** of the following:

- Month-to-month contract
- Tenure under 12 months
- Fiber-optic internet service
- No Online Security and/or no Tech Support
- Electronic check payment method
- 4 or more customer support calls in the recent period

## 5. Retention Recommendations

| Recommendation | Target Segment | Expected Impact |
|---|---|---|
| Contract-upgrade incentives | Month-to-month customers, first 90 days | Reduces highest-churn segment |
| Onboarding intervention program | Tenure 0–6 months | Addresses front-loaded risk window |
| Support-call triage / escalation | Customers on 3rd support call within 60 days | Catches dissatisfaction before it compounds |
| Add-on bundling (free trial) | Fiber / high-bill customers without security/support add-ons | Increases product attachment & switching cost |
| Payment-method nudges | Electronic check payers | Increases payment "stickiness" |

## 6. Suggested Operational Workflow

1. **Score weekly** — run the trained model against the full active customer base every
   week to produce a fresh churn-probability list.
2. **Prioritize by value × risk** — rank flagged customers by churn probability multiplied
   by customer value (MonthlyCharges × Tenure) so retention effort goes where it matters
   most financially.
3. **A/B test offers** — test discount vs. service-upgrade vs. contract-incentive offers
   against a control group to measure true causal lift, not just correlation.
4. **Monitor & retrain** — track precision/recall in production monthly; retrain the model
   quarterly as customer behavior and market conditions shift.

## 7. Limitations & Next Steps

- This dataset is synthetically generated to mirror realistic telecom churn patterns; before
  production use, retrain on the company's actual historical data.
- Recall (74%) leaves roughly 1 in 4 churners unflagged — a higher-recall model (e.g. lowering
  the decision threshold, or an ensemble tuned for recall) could be explored if the cost of a
  missed churner is very high.
- Consider adding customer satisfaction survey scores, usage/data-consumption metrics, or
  competitor pricing data as additional features in a future iteration.
