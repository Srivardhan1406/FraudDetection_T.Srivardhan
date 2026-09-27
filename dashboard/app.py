import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import os

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame({
        "TransactionID"  : range(3000000, 3000000 + n),
        "TransactionAmt" : np.random.exponential(100, n).clip(1, 5000),
        "HourOfDay"      : np.random.randint(0, 24, n),
        "FraudProb"      : np.random.beta(1, 15, n),
        "isFraud"        : np.random.choice([0, 1], n, p=[0.965, 0.035]),
    })
    df.loc[df["isFraud"] == 1, "FraudProb"] = np.random.beta(5, 2, df["isFraud"].sum()).clip(0, 1)

    def risk(p):
        if p >= 0.75:   return "🔴 Critical Risk"
        elif p >= 0.40: return "🟡 Suspicious"
        return "🟢 Clear"

    df["RiskTier"] = df["FraudProb"].apply(risk)
    return df

st.sidebar.title("🛡️ Fraud Ops Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["📊 Overview", "🔍 Transaction Explorer", "🧠 SHAP Explainer"])
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
min_prob    = st.sidebar.slider("Min Fraud Probability", 0.0, 1.0, 0.0, 0.01)
risk_filter = st.sidebar.multiselect("Risk Tier",
    ["🔴 Critical Risk","🟡 Suspicious","🟢 Clear"],
    default=["🔴 Critical Risk","🟡 Suspicious","🟢 Clear"])

df          = load_data()
filtered_df = df[(df["FraudProb"] >= min_prob) & (df["RiskTier"].isin(risk_filter))]

# ── PAGE 1: Overview ──────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 Fraud Detection — Overview")

    total       = len(df)
    fraud_total = df["isFraud"].sum()
    det_rate    = fraud_total / total * 100
    avg_amt     = df[df["isFraud"] == 1]["TransactionAmt"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Total Fraud Detected", f"{int(fraud_total):,}")
    c3.metric("Detection Rate", f"{det_rate:.2f}%")
    c4.metric("Avg Fraud Amount", f"${avg_amt:,.2f}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        tier_counts = df["RiskTier"].value_counts().reset_index()
        tier_counts.columns = ["Risk Tier","Count"]
        fig = px.pie(tier_counts, names="Risk Tier", values="Count", hole=0.5,
                     title="Risk Tier Distribution",
                     color="Risk Tier",
                     color_discrete_map={"🔴 Critical Risk":"#e74c3c",
                                         "🟡 Suspicious":"#f39c12","🟢 Clear":"#2ecc71"})
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        hour_fraud = df[df["isFraud"]==1].groupby("HourOfDay").size().reset_index(name="Count")
        fig2 = px.bar(hour_fraud, x="HourOfDay", y="Count",
                      title="Fraud Count by Hour of Day",
                      color="Count", color_continuous_scale="Reds")
        st.plotly_chart(fig2, use_container_width=True)

    sample = df.sample(min(3000, len(df)), random_state=42)
    fig3 = px.scatter(sample, x="HourOfDay", y="TransactionAmt", color="FraudProb",
                      color_continuous_scale="RdYlGn_r", opacity=0.6,
                      title="Transaction Amount vs Hour — Colored by Fraud Probability")
    st.plotly_chart(fig3, use_container_width=True)

# ── PAGE 2: Transaction Explorer ──────────────────────────────────────────────
elif page == "🔍 Transaction Explorer":
    st.title("🔍 Transaction Explorer")

    search_id = st.text_input("Search by TransactionID", "")
    if search_id:
        result = df[df["TransactionID"].astype(str) == search_id]
        if len(result) > 0:
            row = result.iloc[0]
            st.success("Transaction Found!")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("TransactionID", str(row["TransactionID"]))
            c2.metric("Amount", f"${row['TransactionAmt']:.2f}")
            c3.metric("Fraud Probability", f"{row['FraudProb']:.4f}")
            c4.metric("Risk Tier", row["RiskTier"])
        else:
            st.warning("Transaction ID not found.")

    st.markdown("---")
    st.subheader(f"Showing {len(filtered_df):,} transactions")
    display_cols = ["TransactionID","TransactionAmt","HourOfDay","FraudProb","RiskTier","isFraud"]
    st.dataframe(filtered_df[display_cols].sort_values("FraudProb", ascending=False)
                 .reset_index(drop=True).head(500), use_container_width=True)

    fig4 = px.histogram(filtered_df, x="TransactionAmt", color="RiskTier", log_y=True, nbins=80,
                        title="Transaction Amount Distribution by Risk Tier",
                        color_discrete_map={"🔴 Critical Risk":"#e74c3c",
                                            "🟡 Suspicious":"#f39c12","🟢 Clear":"#2ecc71"})
    st.plotly_chart(fig4, use_container_width=True)

# ── PAGE 3: SHAP Explainer ────────────────────────────────────────────────────
elif page == "🧠 SHAP Explainer":
    st.title("🧠 SHAP Explainer")
    txn_id = st.text_input("TransactionID", placeholder="e.g. 3000042")

    if txn_id:
        row = df[df["TransactionID"].astype(str) == txn_id]
        if len(row) > 0:
            prob = row.iloc[0]["FraudProb"]
            tier = row.iloc[0]["RiskTier"]
            amt  = row.iloc[0]["TransactionAmt"]
            hour = row.iloc[0]["HourOfDay"]

            st.markdown(f"**Fraud Probability:** `{prob:.4f}` | **Risk Tier:** {tier}")

            if prob >= 0.75:
                explanation = (f"⚠️ HIGH FRAUD RISK. Transaction of ${amt:.2f} at hour {hour} "
                               f"flagged with {prob*100:.1f}% fraud probability. "
                               "Key drivers: high amount, late-hour activity, anomalous V-features.")
            elif prob >= 0.40:
                explanation = (f"⚡ SUSPICIOUS. Transaction of ${amt:.2f} at hour {hour} "
                               f"has mixed signals — {prob*100:.1f}% fraud probability. "
                               "Consider additional verification.")
            else:
                explanation = (f"✅ LOW RISK. Transaction of ${amt:.2f} at hour {hour} "
                               f"appears legitimate with only {prob*100:.1f}% fraud probability.")

            st.info(explanation)

            charts_dir = "charts"
            if prob >= 0.75 and os.path.exists(f"{charts_dir}/shap_waterfall_confirmed_fraud.png"):
                st.image(f"{charts_dir}/shap_waterfall_confirmed_fraud.png", use_column_width=True)
            elif prob >= 0.40 and os.path.exists(f"{charts_dir}/shap_waterfall_borderline_case.png"):
                st.image(f"{charts_dir}/shap_waterfall_borderline_case.png", use_column_width=True)
            elif os.path.exists(f"{charts_dir}/shap_waterfall_legitimate_trans.png"):
                st.image(f"{charts_dir}/shap_waterfall_legitimate_trans.png", use_column_width=True)

            if os.path.exists(f"{charts_dir}/shap_summary.png"):
                st.image(f"{charts_dir}/shap_summary.png", caption="Global SHAP Summary", use_column_width=True)
        else:
            st.warning("TransactionID not found. Try a value between 3000000 and 3004999.")

    st.markdown("---")
    st.markdown("**How to read SHAP:** Red = pushes toward fraud. Blue = pushes toward legitimate.")