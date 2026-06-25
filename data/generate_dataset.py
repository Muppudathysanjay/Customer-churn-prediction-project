"""
generate_dataset.py
--------------------
Generates a realistic, synthetic Telecom Customer Churn dataset.

Why synthetic: this lets the project ship with a ready-to-run dataset that
mirrors the structure and statistical relationships of well-known public
churn datasets (e.g., IBM Telco Customer Churn) without any licensing
concerns. Churn probability is built from an underlying logistic model
driven by tenure, contract type, charges, and support-call volume, so all
downstream EDA/ML signal is realistic rather than random noise.

Run:
    python generate_dataset.py
Output:
    customer_churn.csv  (7,043 rows, 18 columns)
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_CUSTOMERS = 7043  # matches the well-known Telco churn dataset size

rng = np.random.default_rng(RANDOM_SEED)


def generate_dataset(n=N_CUSTOMERS) -> pd.DataFrame:
    customer_id = [f"CUST-{10000 + i}" for i in range(n)]

    gender = rng.choice(["Male", "Female"], size=n, p=[0.505, 0.495])

    senior_citizen = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    age = np.where(
        senior_citizen == 1,
        rng.integers(65, 85, size=n),
        rng.integers(18, 65, size=n),
    )

    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])

    # Tenure (months) - skewed so many customers are newer
    tenure = rng.gamma(shape=2.0, scale=18, size=n).astype(int)
    tenure = np.clip(tenure, 0, 72)

    contract_type = rng.choice(
        ["Month-to-Month", "One Year", "Two Year"], size=n, p=[0.55, 0.21, 0.24]
    )

    payment_method = rng.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
        size=n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    paperless_billing = rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41])

    internet_service = rng.choice(
        ["DSL", "Fiber Optic", "No"], size=n, p=[0.34, 0.44, 0.22]
    )

    def dependent_service(base_p_yes, internet_arr):
        out = []
        for svc in internet_arr:
            if svc == "No":
                out.append("No Internet Service")
            else:
                out.append(rng.choice(["Yes", "No"], p=[base_p_yes, 1 - base_p_yes]))
        return np.array(out)

    online_security = dependent_service(0.36, internet_service)
    online_backup = dependent_service(0.39, internet_service)
    device_protection = dependent_service(0.39, internet_service)
    tech_support = dependent_service(0.37, internet_service)
    streaming_tv = dependent_service(0.43, internet_service)
    streaming_movies = dependent_service(0.44, internet_service)

    phone_service = rng.choice(["Yes", "No"], size=n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No Phone Service",
        rng.choice(["Yes", "No"], size=n, p=[0.42, 0.58]),
    )

    # Customer support calls - higher for month-to-month / fiber / low tenure
    base_calls = rng.poisson(lam=1.5, size=n)
    extra_calls = np.where(contract_type == "Month-to-Month", rng.poisson(1.2, n), 0)
    extra_calls += np.where(internet_service == "Fiber Optic", rng.poisson(0.8, n), 0)
    customer_support_calls = np.clip(base_calls + extra_calls, 0, 12)

    # Monthly charges driven by services subscribed
    base_charge = rng.normal(20, 3, n)
    internet_charge = np.select(
        [internet_service == "DSL", internet_service == "Fiber Optic", internet_service == "No"],
        [rng.normal(24, 4, n), rng.normal(45, 6, n), 0],
    )
    addon_charge = (
        (online_security == "Yes").astype(int) * rng.normal(5, 1, n)
        + (online_backup == "Yes").astype(int) * rng.normal(5, 1, n)
        + (device_protection == "Yes").astype(int) * rng.normal(5, 1, n)
        + (tech_support == "Yes").astype(int) * rng.normal(5, 1, n)
        + (streaming_tv == "Yes").astype(int) * rng.normal(8, 1, n)
        + (streaming_movies == "Yes").astype(int) * rng.normal(8, 1, n)
        + (multiple_lines == "Yes").astype(int) * rng.normal(7, 1, n)
    )
    monthly_charges = np.round(np.clip(base_charge + internet_charge + addon_charge, 18, 130), 2)

    total_charges = np.round(monthly_charges * tenure * rng.uniform(0.94, 1.0, n) + rng.normal(0, 5, n), 2)
    total_charges = np.clip(total_charges, 0, None)
    # New customers (tenure=0) should have ~ total = 1 month or small value
    total_charges = np.where(tenure == 0, np.round(monthly_charges * rng.uniform(0.9, 1.0, n), 2), total_charges)

    # ---- Underlying churn probability model (logistic) ----
    z = (
        -1.6
        + 0.032 * customer_support_calls ** 1.3
        - 0.045 * tenure
        + 0.012 * monthly_charges
        + np.where(contract_type == "Month-to-Month", 1.05, 0)
        + np.where(contract_type == "One Year", 0.25, 0)
        + np.where(contract_type == "Two Year", -0.55, 0)
        + np.where(internet_service == "Fiber Optic", 0.35, 0)
        + np.where(payment_method == "Electronic Check", 0.40, 0)
        + np.where(online_security == "No", 0.28, 0)
        + np.where(tech_support == "No", 0.22, 0)
        + np.where(paperless_billing == "Yes", 0.18, 0)
        + np.where(senior_citizen == 1, 0.15, 0)
        + np.where(partner == "No", 0.12, 0)
        + np.where(dependents == "No", 0.10, 0)
        + rng.normal(0, 0.55, n)  # noise
    )
    churn_prob = 1 / (1 + np.exp(-z))
    churn = (rng.uniform(0, 1, n) < churn_prob).astype(int)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame(
        {
            "CustomerID": customer_id,
            "Gender": gender,
            "SeniorCitizen": senior_citizen,
            "Age": age,
            "Partner": partner,
            "Dependents": dependents,
            "Tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract_type,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "CustomerSupportCalls": customer_support_calls,
            "Churn": churn_label,
        }
    )

    # Inject a small amount of realistic messiness for the cleaning step to handle.
    # Cast to object first so we can mix floats, NaN, and whitespace strings
    # (a very common real-world artifact in exported billing data).
    df["TotalCharges"] = df["TotalCharges"].astype(object)

    missing_idx = rng.choice(df.index, size=int(0.012 * n), replace=False)
    df.loc[missing_idx, "TotalCharges"] = np.nan

    blank_idx = rng.choice(df.index, size=int(0.004 * n), replace=False)
    df.loc[blank_idx, "TotalCharges"] = " "  # whitespace string, common real-world artifact

    dup_rows = df.sample(n=15, random_state=RANDOM_SEED)
    df = pd.concat([df, dup_rows], ignore_index=True)

    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


if __name__ == "__main__":
    data = generate_dataset()
    data.to_csv("customer_churn.csv", index=False)
    print(f"Saved customer_churn.csv with shape {data.shape}")
    print(data["Churn"].value_counts(normalize=True))
