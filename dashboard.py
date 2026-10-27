import streamlit as st
import pandas as pd
import plotly.express as px
import psycopg2

def show():
    # -----------------------------------
    # PAGE CONFIG
    # -----------------------------------
    # st.set_page_config(
    #     page_title="Motor Insurance Executive Dashboard",
    #     page_icon="🚗",
    #     layout="wide"
    # )

    # st.markdown(
    #     """
    #     <style>
    #     .metric-card {
    #         background-color: #f8f9fa;
    #         padding: 15px;
    #         border-radius: 12px;
    #         box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )

    # -----------------------------------
    # DATABASE CONNECTION
    # -----------------------------------
    DB_CONFIG = {
        "dbname": "AutoMotor_Insurance",
        "user": "postgres",
        "password": "United2025",
        "host": "localhost",
        "port": "5432"
    }

    @st.cache_data
    def load_data():
        conn = psycopg2.connect(**DB_CONFIG)
        premium_df = pd.read_sql("SELECT * FROM premium_results", conn)
        risk_df = pd.read_sql("SELECT * FROM risk_profile_results", conn)
        dataset_df = pd.read_sql("SELECT * FROM motor_insurance_data", conn)
        conn.close()
        return premium_df, risk_df, dataset_df

    premium_df, risk_df, dataset_df = load_data()

    # -----------------------------------
    # HEADER
    # -----------------------------------
    st.title("🚗 Motor Insurance Analytics Dashboard")
    st.markdown("### Real-Time Risk & Premium Intelligence")

    # -----------------------------------
    # KPI SECTION
    # -----------------------------------
    total_premium = premium_df["predicted_premium"].sum()
    total_policies = len(premium_df)
    avg_premium = premium_df["predicted_premium"].mean()
    high_risk_pct = (
        (risk_df["risk_label"].str.lower() == "high").sum() / len(risk_df) * 100
        if not risk_df.empty else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Premium", f"{total_premium:,.0f}")
    col2.metric("📄 Total Policies", f"{total_policies:,}")
    col3.metric("📊 Avg Premium", f"{avg_premium:,.0f}")
    col4.metric("⚠️ High Risk %", f"{high_risk_pct:.1f}%")

    st.divider()

    # -----------------------------------
    # MONTHLY TREND & RISK DISTRIBUTION (2 graphs in 1 row)
    # -----------------------------------
    col1, col2 = st.columns(2)

    # Monthly Trend
    if "created_at" in premium_df.columns:
        premium_df["month"] = pd.to_datetime(premium_df["created_at"]).dt.to_period("M")
        monthly = premium_df.groupby("month")["predicted_premium"].sum().reset_index()
        monthly["month"] = monthly["month"].astype(str)

        fig_trend = px.line(
            monthly,
            x="month",
            y="predicted_premium",
            markers=True,
            title="📈 Monthly Premium Trend",
            template="plotly_white"
        )
        col1.plotly_chart(fig_trend, use_container_width=True)

    # Risk Distribution
    if not risk_df.empty:
        risk_counts = risk_df["risk_label"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]

        fig_risk = px.pie(
            risk_counts,
            names="Risk Level",
            values="Count",
            hole=0.4,
            title="Risk Distribution",
            template="plotly_white"
        )
        col2.plotly_chart(fig_risk, use_container_width=True)

    # -----------------------------------
    # VEHICLE BODY VS PREMIUM & DRIVER AGE DISTRIBUTION (2 graphs in 1 row)
    # -----------------------------------
    col1, col2 = st.columns(2)

    # Vehicle Body vs Premium
    if "vehicle_body_type" in dataset_df.columns:
        body_premium = dataset_df.groupby("vehicle_body_type")["sum_insured"].mean().reset_index()

        fig_body = px.bar(
            body_premium,
            x="vehicle_body_type",
            y="sum_insured",
            title="Average Sum Insured by Vehicle Body Type",
            template="plotly_white"
        )
        col1.plotly_chart(fig_body, use_container_width=True)

    # Driver Age Distribution
    if "driver_age" in dataset_df.columns and "predicted_rate" in premium_df.columns:
        age_data = dataset_df.copy()
        age_data["Age Group"] = pd.cut(
            age_data["driver_age"],
            bins=[18,25,35,50,100],
            labels=["18-25","26-35","36-50","51+"]
        )

        fig_age = px.box(
            age_data,
            x="Age Group",
            y="driver_age",
            title="Driver Age Distribution",
            template="plotly_white"
        )
        col2.plotly_chart(fig_age, use_container_width=True)

    # -----------------------------------
    # TOP 10 HIGH PREMIUM POLICIES (2 graphs in 1 row)
    # -----------------------------------
    st.subheader("🏆 Top 10 Highest Premium Policies")

    top10 = premium_df.sort_values(
        by="predicted_premium",
        ascending=False
    ).head(10)

    # Create label column
    top10['policy_label'] = top10['vehicle_make'] + ' ' + top10['vehicle_model'] + ' (' + top10['vehicle_make_year'].astype(str) + ')'

    col1, col2 = st.columns(2)

    # Bar chart
    fig_top10 = px.bar(
        top10,
        x="policy_label",
        y="predicted_premium",
        color="predicted_premium",
        text="predicted_premium",
        title="Top 10 Highest Predicted Premium Policies",
        template="plotly_white"
    )
    fig_top10.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    col1.plotly_chart(fig_top10, use_container_width=True)

    # Treemap
    fig_treemap = px.treemap(
        top10,
        path=["policy_label"],
        values="predicted_premium",
        color="predicted_premium",
        color_continuous_scale="Viridis",
        title="Top 10 Predicted Premium Policies Treemap"
    )
    col2.plotly_chart(fig_treemap, use_container_width=True)

    # -----------------------------------
    # FOOTER
    # -----------------------------------
    st.markdown("---")
    st.caption("Developed for Motor Insurance Risk & Premium Intelligence System")


# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    show()