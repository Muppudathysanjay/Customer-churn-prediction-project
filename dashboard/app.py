"""
app.py
------
Customer Churn Prediction — Interactive Streamlit Dashboard

A production-style, single-file Streamlit app with:
- KPI cards summarizing the customer base
- Churn overview with interactive Plotly charts
- A real-time churn prediction form (risk score + segment)
- Model performance comparison (loaded from training artifacts)
- Business recommendations
- CSV export of predictions

Run with:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG  (must be the first Streamlit call)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "customer_churn_cleaned.csv"
RAW_DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"
MODEL_PATH = BASE_DIR / "models" / "customer_churn_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.pkl"
NUMERIC_FEATURES_PATH = BASE_DIR / "models" / "numeric_features.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

# ----------------------------------------------------------------------------
# THEME / STYLE
# Palette chosen deliberately for a churn dashboard: a calm slate/navy base
# with a single warning-red accent reserved ONLY for churn-risk signals, and
# a teal accent for "healthy / retained" signals. This keeps risk legible
# at a glance instead of defaulting to a generic blue-everything dashboard.
# ----------------------------------------------------------------------------
COLOR_SAFE = "#1E8E6E"      # teal-green: retained / healthy
COLOR_RISK = "#D64550"      # warning red: churn / risk
COLOR_PRIMARY = "#243B53"   # deep slate navy: structural / primary
COLOR_ACCENT = "#F0A04B"    # warm amber: secondary highlight
COLOR_MUTED = "#8896A6"     # muted slate: gridlines / secondary text

PLOTLY_TEMPLATE = "plotly_white"

CUSTOM_CSS = f"""
<style>
    .stApp {{
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.85rem;
        font-weight: 700;
        color: {COLOR_PRIMARY};
    }}
    [data-testid="stMetricLabel"] {{
        font-weight: 600;
        color: {COLOR_MUTED};
    }}
    .kpi-card {{
        background-color: #FFFFFF;
        border: 1px solid #E3E8EE;
        border-left: 5px solid {COLOR_PRIMARY};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .risk-badge-high {{
        background-color: {COLOR_RISK};
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }}
    .risk-badge-medium {{
        background-color: {COLOR_ACCENT};
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }}
    .risk-badge-low {{
        background-color: {COLOR_SAFE};
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }}
    .section-divider {{
        border-top: 1px solid #E3E8EE;
        margin: 1.5rem 0 1.5rem 0;
    }}
    h1, h2, h3 {{
        color: {COLOR_PRIMARY};
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# DATA & MODEL LOADING  (cached so the dashboard stays fast)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the cleaned, feature-engineered dataset used for EDA + KPI displays."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_csv(RAW_DATA_PATH)
    return df


@st.cache_resource
def load_model_artifacts():
    """Load the trained model, scaler, and feature column order."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    feature_columns = joblib.load(FEATURES_PATH) if FEATURES_PATH.exists() else None
    numeric_features = joblib.load(NUMERIC_FEATURES_PATH) if NUMERIC_FEATURES_PATH.exists() else None
    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
    return model, scaler, feature_columns, numeric_features, metadata


df = load_data()
model, scaler, feature_columns, numeric_features, metadata = load_model_artifacts()
needs_scaling = metadata.get("needs_scaling", False)
best_model_name = metadata.get("best_model_name", model.__class__.__name__)


# ----------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------
def build_feature_row(inputs: dict) -> pd.DataFrame:
    """
    Convert a dict of human-friendly form inputs into a single-row dataframe
    matching the exact column structure the trained model expects
    (same encoding + engineered features used in the notebook).
    """
    row = {
        "Gender": 1 if inputs["gender"] == "Male" else 0,
        "SeniorCitizen": 1 if inputs["age"] >= 65 else 0,
        "Age": inputs["age"],
        "Partner": 1 if inputs["partner"] == "Yes" else 0,
        "Dependents": 1 if inputs["dependents"] == "Yes" else 0,
        "Tenure": inputs["tenure"],
        "PhoneService": 1 if inputs["phone_service"] == "Yes" else 0,
        "PaperlessBilling": 1 if inputs["paperless_billing"] == "Yes" else 0,
        "MonthlyCharges": inputs["monthly_charges"],
        "TotalCharges": inputs["monthly_charges"] * inputs["tenure"],
        "CustomerSupportCalls": inputs["support_calls"],
    }

    # Engineered features (must mirror notebook section 5 exactly)
    row["AvgChargePerTenure"] = row["TotalCharges"] / max(row["Tenure"], 1)
    row["IsNewCustomer"] = 1 if row["Tenure"] <= 6 else 0
    row["HighSupportCalls"] = 1 if row["CustomerSupportCalls"] >= 4 else 0

    addon_cols_yes = [
        inputs["online_security"], inputs["online_backup"], inputs["device_protection"],
        inputs["tech_support"], inputs["streaming_tv"], inputs["streaming_movies"],
    ]
    row["TotalAddonServices"] = sum(1 for v in addon_cols_yes if v == "Yes")
    row["IsMonthToMonth"] = 1 if inputs["contract"] == "Month-to-Month" else 0

    # One-hot categorical placeholders, filled in below
    base_df = pd.DataFrame([row])

    one_hot_specs = {
        "MultipleLines": (inputs["multiple_lines"], ["Yes", "No Phone Service"]),
        "InternetService": (inputs["internet_service"], ["Fiber Optic", "No"]),
        "OnlineSecurity": (inputs["online_security"], ["No", "No Internet Service"]),
        "OnlineBackup": (inputs["online_backup"], ["No", "No Internet Service"]),
        "DeviceProtection": (inputs["device_protection"], ["No", "No Internet Service"]),
        "TechSupport": (inputs["tech_support"], ["No", "No Internet Service"]),
        "StreamingTV": (inputs["streaming_tv"], ["No", "No Internet Service"]),
        "StreamingMovies": (inputs["streaming_movies"], ["No", "No Internet Service"]),
        "Contract": (inputs["contract"], ["Month-to-Month", "One Year", "Two Year"]),
        "PaymentMethod": (inputs["payment_method"], ["Credit Card", "Electronic Check", "Mailed Check"]),
    }
    for col_prefix, (value, categories) in one_hot_specs.items():
        for cat in categories:
            colname = f"{col_prefix}_{cat}"
            base_df[colname] = 1 if value == cat else 0

    # ChargeVsTierAvg: approximate using global tier averages from the dashboard dataset
    tier_avg_map = df.groupby("InternetService")["MonthlyCharges"].mean().to_dict()
    tier_avg = tier_avg_map.get(inputs["internet_service"], df["MonthlyCharges"].mean())
    base_df["ChargeVsTierAvg"] = row["MonthlyCharges"] - tier_avg

    # Align exactly to the training feature order, filling any unseen column with 0
    if feature_columns is not None:
        for col in feature_columns:
            if col not in base_df.columns:
                base_df[col] = 0
        base_df = base_df[feature_columns]

    return base_df


def predict_churn(inputs: dict):
    """Run the full pipeline (build features -> scale if needed -> predict) for one customer."""
    X_row = build_feature_row(inputs)
    X_for_model = X_row.copy()
    if needs_scaling and scaler is not None and numeric_features is not None:
        X_for_model[numeric_features] = scaler.transform(X_row[numeric_features])
    proba = model.predict_proba(X_for_model)[0, 1]
    pred = int(proba >= 0.5)
    return pred, proba


def risk_tier(proba: float) -> tuple:
    if proba >= 0.6:
        return "High Risk", "risk-badge-high", COLOR_RISK
    elif proba >= 0.3:
        return "Medium Risk", "risk-badge-medium", COLOR_ACCENT
    else:
        return "Low Risk", "risk-badge-low", COLOR_SAFE


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 Churn Prediction")
    st.caption("Customer Churn Prediction — Data Science Project")

    theme_choice = st.radio("Appearance", ["Light", "Dark"], horizontal=True, index=0)

    st.markdown("---")
    st.markdown("### Filters")
    contract_filter = st.multiselect(
        "Contract Type", options=sorted(df["Contract"].unique()), default=list(df["Contract"].unique())
    )
    internet_filter = st.multiselect(
        "Internet Service", options=sorted(df["InternetService"].unique()), default=list(df["InternetService"].unique())
    )
    tenure_range = st.slider("Tenure (months)", int(df["Tenure"].min()), int(df["Tenure"].max()),
                              (int(df["Tenure"].min()), int(df["Tenure"].max())))

    st.markdown("---")
    st.markdown(f"**Active Model:** {best_model_name}")
    if metadata.get("metrics"):
        auc = metadata["metrics"].get(best_model_name, {}).get("ROC-AUC")
        if auc:
            st.markdown(f"**Test ROC-AUC:** {auc:.3f}")
    st.caption("Built with scikit-learn, XGBoost & Streamlit")

# Dark theme override (applied after sidebar so the toggle takes effect this run)
if theme_choice == "Dark":
    st.markdown(
        """
        <style>
            .stApp { background-color: #0E1117; color: #E3E8EE; }
            .kpi-card { background-color: #1A1F29 !important; border-color: #2A2F3A !important; color: #E3E8EE; }
            h1, h2, h3, p, label, span { color: #E3E8EE !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Apply filters to a working copy used across tabs
filtered_df = df[
    df["Contract"].isin(contract_filter)
    & df["InternetService"].isin(internet_filter)
    & df["Tenure"].between(tenure_range[0], tenure_range[1])
].copy()

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown(
    "Analyze customer behavior, identify churn drivers, score individual customers in "
    "real time, and act on data-backed retention recommendations."
)

# ----------------------------------------------------------------------------
# KPI CARDS
# ----------------------------------------------------------------------------
total_customers = len(filtered_df)
churned_customers = (filtered_df["Churn"] == "Yes").sum()
churn_rate = churned_customers / total_customers * 100 if total_customers else 0
avg_monthly_charge = filtered_df["MonthlyCharges"].mean() if total_customers else 0
avg_tenure = filtered_df["Tenure"].mean() if total_customers else 0
monthly_revenue_at_risk = filtered_df.loc[filtered_df["Churn"] == "Yes", "MonthlyCharges"].sum()

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Customers", f"{total_customers:,}")
kpi2.metric("Churn Rate", f"{churn_rate:.1f}%", delta=f"{churned_customers:,} churned", delta_color="inverse")
kpi3.metric("Avg. Monthly Charges", f"${avg_monthly_charge:,.2f}")
kpi4.metric("Avg. Tenure", f"{avg_tenure:.1f} mo")
kpi5.metric("Monthly Revenue at Risk", f"${monthly_revenue_at_risk:,.0f}")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab_overview, tab_predict, tab_performance, tab_insights = st.tabs(
    ["📈 Churn Overview", "🔮 Predict Churn", "🧪 Model Performance", "💡 Business Insights"]
)

# ============================== TAB 1: OVERVIEW ==============================
with tab_overview:
    st.subheader("Customer Churn Overview")

    c1, c2 = st.columns(2)
    with c1:
        churn_counts = filtered_df["Churn"].value_counts().reindex(["No", "Yes"]).fillna(0)
        fig = px.pie(
            names=churn_counts.index, values=churn_counts.values,
            color=churn_counts.index, color_discrete_map={"No": COLOR_SAFE, "Yes": COLOR_RISK},
            title="Churn Distribution", hole=0.45, template=PLOTLY_TEMPLATE,
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        contract_churn = (
            filtered_df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
        )
        contract_churn.columns = ["Contract", "ChurnRate"]
        fig = px.bar(
            contract_churn.sort_values("ChurnRate", ascending=False), x="Contract", y="ChurnRate",
            color="ChurnRate", color_continuous_scale=["#1E8E6E", "#F0A04B", "#D64550"],
            title="Churn Rate (%) by Contract Type", text_auto=".1f", template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(yaxis_title="Churn Rate (%)", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(
            filtered_df, x="Churn", y="MonthlyCharges", color="Churn",
            color_discrete_map={"No": COLOR_SAFE, "Yes": COLOR_RISK},
            title="Monthly Charges by Churn Status", template=PLOTLY_TEMPLATE,
            category_orders={"Churn": ["No", "Yes"]},
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        payment_churn = (
            filtered_df.groupby("PaymentMethod")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
        )
        payment_churn.columns = ["PaymentMethod", "ChurnRate"]
        fig = px.bar(
            payment_churn.sort_values("ChurnRate", ascending=False), x="PaymentMethod", y="ChurnRate",
            color="ChurnRate", color_continuous_scale=["#1E8E6E", "#F0A04B", "#D64550"],
            title="Churn Rate (%) by Payment Method", text_auto=".1f", template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(yaxis_title="Churn Rate (%)", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Tenure vs. Churn")
    tenure_bins = [0, 12, 24, 48, 72]
    tenure_labels = ["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"]
    tmp = filtered_df.copy()
    tmp["TenureGroup"] = pd.cut(tmp["Tenure"], bins=tenure_bins, labels=tenure_labels, include_lowest=True)
    tenure_churn = tmp.groupby("TenureGroup")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
    tenure_churn.columns = ["TenureGroup", "ChurnRate"]
    fig = px.line(
        tenure_churn, x="TenureGroup", y="ChurnRate", markers=True,
        title="Churn Rate (%) by Tenure Group", template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(line_color=COLOR_RISK, marker=dict(size=10, color=COLOR_PRIMARY))
    fig.update_layout(yaxis_title="Churn Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Customer Segmentation (Tenure vs. Monthly Charges)")
    sample_df = filtered_df.sample(min(2000, len(filtered_df)), random_state=42) if len(filtered_df) else filtered_df
    fig = px.scatter(
        sample_df, x="Tenure", y="MonthlyCharges", color="Churn",
        color_discrete_map={"No": COLOR_SAFE, "Yes": COLOR_RISK},
        opacity=0.55, title="Customer Distribution: Tenure vs. Monthly Charges",
        template=PLOTLY_TEMPLATE, category_orders={"Churn": ["No", "Yes"]},
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 View Raw / Filtered Data"):
        st.dataframe(filtered_df, use_container_width=True, height=300)
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Filtered Data as CSV", data=csv_data,
            file_name="filtered_customer_data.csv", mime="text/csv",
        )

# ============================== TAB 2: PREDICT ==============================
with tab_predict:
    st.subheader("Real-Time Churn Prediction")
    st.markdown("Fill in a customer's profile to get an instant churn risk score.")

    with st.form("prediction_form"):
        f1, f2, f3 = st.columns(3)
        with f1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 90, 35)
            partner = st.selectbox("Has Partner?", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        with f2:
            contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
            payment_method = st.selectbox(
                "Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"]
            )
            paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 130.0, 65.0, step=0.5)
            support_calls = st.slider("Customer Support Calls (last period)", 0, 12, 1)
        with f3:
            phone_service = st.selectbox("Phone Service?", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines?", ["Yes", "No", "No Phone Service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No Internet Service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No Internet Service"])

        f4, f5 = st.columns(2)
        with f4:
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No Internet Service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No Internet Service"])
        with f5:
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No Internet Service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No Internet Service"])

        submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)

    if submitted:
        inputs = dict(
            gender=gender, age=age, partner=partner, dependents=dependents, tenure=tenure,
            contract=contract, payment_method=payment_method, paperless_billing=paperless_billing,
            monthly_charges=monthly_charges, support_calls=support_calls, phone_service=phone_service,
            multiple_lines=multiple_lines, internet_service=internet_service,
            online_security=online_security, online_backup=online_backup,
            device_protection=device_protection, tech_support=tech_support,
            streaming_tv=streaming_tv, streaming_movies=streaming_movies,
        )
        pred, proba = predict_churn(inputs)
        tier_label, tier_class, tier_color = risk_tier(proba)

        st.markdown("---")
        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(f"<span class='{tier_class}'>{tier_label}</span>", unsafe_allow_html=True)
            st.metric("Churn Probability", f"{proba*100:.1f}%")
            st.metric("Predicted Outcome", "Will Churn" if pred == 1 else "Will Stay")

        with r2:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    number={"suffix": "%"},
                    title={"text": "Churn Risk Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": tier_color},
                        "steps": [
                            {"range": [0, 30], "color": "#DFF3EC"},
                            {"range": [30, 60], "color": "#FCEBD5"},
                            {"range": [60, 100], "color": "#FBE0E2"},
                        ],
                        "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": 50},
                    },
                )
            )
            fig.update_layout(height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # Tailored recommendation based on risk tier and inputs
        st.markdown("##### Suggested Action")
        if tier_label == "High Risk":
            actions = []
            if contract == "Month-to-Month":
                actions.append("offer an incentive to upgrade to a 1- or 2-year contract")
            if support_calls >= 4:
                actions.append("escalate immediately to a retention specialist (high support-call volume)")
            if online_security == "No" or tech_support == "No":
                actions.append("offer a free trial of Online Security / Tech Support to increase engagement")
            if payment_method == "Electronic Check":
                actions.append("nudge toward autopay/credit card with a small bill credit")
            if not actions:
                actions.append("proactively reach out with a loyalty offer")
            st.error("**Immediate retention action recommended:** " + "; ".join(actions) + ".")
        elif tier_label == "Medium Risk":
            st.warning("**Monitor closely:** consider a check-in or value-reinforcement message in the next billing cycle.")
        else:
            st.success("**Low risk:** no action needed beyond standard engagement.")

        # Export this single prediction
        result_row = pd.DataFrame([{**inputs, "ChurnProbability": round(proba, 4), "PredictedChurn": "Yes" if pred else "No"}])
        st.download_button(
            "⬇️ Export This Prediction as CSV",
            data=result_row.to_csv(index=False).encode("utf-8"),
            file_name="churn_prediction_result.csv", mime="text/csv",
        )

# ============================== TAB 3: PERFORMANCE ==============================
with tab_performance:
    st.subheader("Model Performance & Comparison")

    if metadata.get("metrics"):
        metrics_df = pd.DataFrame(metadata["metrics"]).T
        metrics_df.index.name = "Model"
        metrics_df = metrics_df.sort_values("ROC-AUC", ascending=False).round(4)

        st.dataframe(
            metrics_df.style.highlight_max(axis=0, color="#DFF3EC"),
            use_container_width=True,
        )

        fig = go.Figure()
        for metric in metrics_df.columns:
            fig.add_trace(go.Bar(name=metric, x=metrics_df.index, y=metrics_df[metric]))
        fig.update_layout(
            barmode="group", title="Model Comparison Across Metrics",
            yaxis_title="Score", template=PLOTLY_TEMPLATE, legend_title="Metric",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"🏆 **Best Model Selected: {best_model_name}** (highest ROC-AUC, the most robust metric under class imbalance)")
    else:
        st.info("Run the training notebook to generate model_metadata.json with comparison metrics.")

    st.markdown("##### Feature Importance")
    if metadata.get("top_features"):
        feat_imp = pd.Series(metadata["top_features"]).sort_values(ascending=True)
        fig = px.bar(
            x=feat_imp.values, y=feat_imp.index, orientation="h",
            title="Top Features Driving Churn Predictions",
            color=feat_imp.values, color_continuous_scale="Reds", template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(xaxis_title="Importance Score", yaxis_title="", coloraxis_showscale=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available in metadata.")

# ============================== TAB 4: INSIGHTS ==============================
with tab_insights:
    st.subheader("Business Insights & Retention Strategy")

    st.markdown("""
##### 🔎 Why Customers Leave
1. **Short-term commitment** — Month-to-month customers churn far more than contract customers.
2. **Early-lifecycle risk** — Churn is concentrated in the first 6–12 months of tenure.
3. **Service friction** — 4+ support calls is a strong early-warning signal of dissatisfaction.
4. **Price sensitivity at the high end** — Fiber-optic / high-bill customers churn more.
5. **Payment friction** — Electronic check payers churn more than autopay customers.
6. **Low product attachment** — Customers without security/support add-ons churn more easily.
""")

    st.markdown("##### 🎯 High-Risk Customer Profile")
    st.info(
        "Month-to-month contract • Tenure < 12 months • Fiber-optic internet • "
        "No online security / tech support • Electronic check payment • 4+ recent support calls"
    )

    st.markdown("""
##### 🛡️ Retention Recommendations
- **Incentivize contract upgrades** for month-to-month customers, especially in their first 90 days.
- **Onboarding intervention program** with proactive check-ins during months 1–6.
- **Support-call triage**: flag customers on their 3rd call within 60 days for specialist follow-up.
- **Bundle add-ons**: free trials of Online Security / Tech Support for fiber and high-bill customers.
- **Payment method nudges**: incentivize a switch from electronic check to autopay.

##### 📈 Suggested Operational Strategy
1. Score the full customer base weekly using the trained model.
2. Route high-risk customers into a retention workflow, prioritized by churn probability **and** customer value.
3. A/B test retention offers against a control group to measure causal impact.
4. Track precision/recall in production monthly; retrain quarterly as behavior shifts.
""")

    st.markdown("##### 📄 Download Full Report")
    report_path = BASE_DIR / "reports" / "Customer_Churn_Business_Report.md"
    if report_path.exists():
        with open(report_path, "rb") as f:
            st.download_button(
                "⬇️ Download Business Report (Markdown)", data=f, file_name="Customer_Churn_Business_Report.md",
            )
    else:
        st.caption("Business report will appear here once generated.")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption("Customer Churn Prediction Dashboard · Built with Streamlit, scikit-learn & XGBoost · For portfolio / educational use")
