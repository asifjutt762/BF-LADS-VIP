import streamlit as st
import pandas as pd
import joblib
import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import risk_profile
import premium


# ==============================
# 🔧 Database Configuration
# ==============================
DB_USER = "postgres"
DB_PASSWORD = "United2025"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "AutoMotor_Insurance"

MODEL_DIR = "models"
TRAINING_RESULTS_TABLE = "model_training_results"
RISK_RESULTS_TABLE = "risk_profile_results"
PREMIUM_RESULTS_TABLE = "premium_results"

# NEW TABLE NAME
DB_VEHICLE_TABLE = "motor_vehicle"


# ==============================
# 🔗 MODEL FILE MAPPING
# ==============================
MODEL_FILE_MAP = {
    "GradientBoost Model": "GradientBoost_Model.pkl",
    "LightGBoost Model": "LightGBoost_Model.pkl",
    "XGBoost Model": "XGBoost_Model.pkl",
    "CatBoost Model": "CatBoost_Model.pkl",
    "AdaBoost Model": "AdaBoost_Model.pkl"
}


# ==============================
# 📡 Database Connection
# ==============================
def get_db_connection():
    try:
        return create_engine(
            f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
    except SQLAlchemyError as e:
        st.error(f"❌ Database Connection Failed: {e}")
        return None


# =====================================================
# 🆕 NEW: LOAD VEHICLE MAKE & MODEL FROM motor_vehicle
# =====================================================
def load_vehicle_table(engine):
    try:
        df = pd.read_sql("SELECT DISTINCT vehicle_make, vehicle_model FROM motor_vehicle;", engine)

        return df
    except Exception as e:
        st.error(f"❌ Could not load motor_vehicle table: {e}")
        return pd.DataFrame(columns=["vehicle_make", "vehicle_model"])


# ==============================
# 🎯 MAIN SCREEN
# ==============================
def show():

    st.subheader("📊 Model Validation & Selection")

    engine = get_db_connection()
    if engine is None:
        return

    # --------------------------------------------------------
    # 🚀 LOAD TRAINING RESULTS
    # --------------------------------------------------------
    try:
        df_models = pd.read_sql(
            f"SELECT model_name,r2_score FROM {TRAINING_RESULTS_TABLE} where model_name not in ('AdaBoost Model');", # where accuracy in (select max(accuracy) from {TRAINING_RESULTS_TABLE} )
            engine
        )

        if df_models.empty:
            st.warning("⚠️ No trained models found.")
            return

        st.subheader("📄 Trained Models Summary")
        st.dataframe(df_models, use_container_width=True)

    except SQLAlchemyError as e:
        st.error(f"❌ Could not fetch training results: {e}")
        return
    finally:
        engine.dispose()

    # --------------------------------------------------------
    # 🎛 MODEL SELECTION (manual)
    # --------------------------------------------------------
  
    # model_list = df_models["model_name"].tolist()

    # selected_model = st.selectbox(
    #     "Select Model",
    #     options=model_list
    # )

    # # Display accuracy
    # acc_map = dict(zip(df_models["model_name"], df_models["r2_score"]))
    # #st.metric("R² Score", f"{acc_map[selected_model]:.2f}%")
    # st.metric("R² Score", f"{acc_map[selected_model] * 100:.2f}%")

    model_list = df_models["model_name"].tolist()

    col1, col2 = st.columns(2)

    with col1:
        selected_model = st.selectbox(
            "Select Model",
            options=model_list
        )

    # Map model to R² score
    acc_map = dict(zip(df_models["model_name"], df_models["r2_score"]))

    with col2:
        st.metric(
            "R² Score",
            f"{acc_map[selected_model] * 100:.2f}%"
        )

    normalized_model = selected_model.replace("LightGBM", "LightGBoost")

    if normalized_model not in MODEL_FILE_MAP:
        st.error(f"❌ Model missing mapping: {normalized_model}")
        return

    # --------------------------------------------------------
    # 📦 LOAD MODEL FILE
    # --------------------------------------------------------
    model_path = os.path.join(MODEL_DIR, MODEL_FILE_MAP[normalized_model])

    if not os.path.exists(model_path):
        st.error(f"🚫 Model File Missing: {model_path}")
        return

    loaded_model = joblib.load(model_path)
    st.session_state["loaded_model"] = loaded_model
    st.session_state["selected_model_name"] = normalized_model

    #st.success(f"📌 Loaded Model: {os.path.basename(model_path)}")

    # --------------------------------------------------------
    # 📦 LOAD FEATURES FILE
    # --------------------------------------------------------
    features_path = model_path.replace(".pkl", "_features.pkl")

    # if not os.path.exists(features_path):
    #     st.session_state["model_features"] = joblib.load(features_path)
    #     st.info("✅ Feature Set Loaded Successfully")
    # else:
    if not os.path.exists(features_path):
        st.session_state["model_features"] = None
        st.warning("⚠ Feature file not found.")

    # ============================================================
    # 🆕 LOAD VEHICLE MAKE & MODEL TABLE INTO SESSION
    # ============================================================
    try:
        engine = get_db_connection()
        vehicle_df = load_vehicle_table(engine)

        # 🔥 Store vehicle table for use inside risk_profile.show()
        st.session_state["vehicle_df"] = vehicle_df

    except:
        st.error("❌ Failed loading vehicle table.")
    finally:
        if engine:
            engine.dispose()

    # --------------------------------------------------------
    # ACTION BUTTONS
    # --------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔮 Predict Premium"):
            st.session_state["show_prediction"] = True
            st.session_state["show_results_screen"] = False

    with col2:
        if st.button("📑 View Stored Results"):
            st.session_state["show_results_screen"] = True
            st.session_state["show_prediction"] = False

    # --------------------------------------------------------
    # 📑 SHOW RESULTS (risk + premium)
    # --------------------------------------------------------
    if st.session_state.get("show_results_screen", False):
        st.markdown("---")
        st.subheader("📑 All Stored Results")

        try:
            engine = get_db_connection()

            df_risk = pd.read_sql(f"SELECT * FROM {RISK_RESULTS_TABLE}", engine)
            df_premium = pd.read_sql(f"SELECT * FROM {PREMIUM_RESULTS_TABLE}", engine)

             # ✅ FIX: Make both id columns same datatype
            df_risk["id"] = df_risk["id"].astype(str)
            df_premium["id"] = df_premium["id"].astype(str)


            # df_combined = pd.concat([df_risk, df_premium], axis=1)
            # df_combined = pd.merge(df_risk, df_premium, on="id", suffixes=("_risk", "_premium"))
            df_combined = pd.merge(
            df_risk,
            df_premium,
            on="id",
            suffixes=("_risk", "_premium")
                )
            st.dataframe(df_combined, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Could not load results: {e}")
        finally:
            engine.dispose()

    # --------------------------------------------------------
    # 🚦 RUN: RISK PROFILE + PREMIUM
    # --------------------------------------------------------
    if st.session_state.get("show_prediction", False):

        st.markdown("---")
        st.subheader("🚦 Step 1: Risk Profile Prediction")

        # risk_profile.show() will now use st.session_state["vehicle_df"]
        risk_profile.show()

        st.markdown("---")
        st.subheader("💰 Step 2: Premium Prediction")

        premium.show()
