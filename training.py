# training_streamlit_merged.py
"""
Unified Streamlit training dashboard
Combines CatBoost, XGBoost, LightGBM, GradientBoosting, AdaBoost training logic with LLM explanations
Saves models & features to local `models/` and training results to PostgreSQL.

Run: streamlit run training_streamlit_merged.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import requests
import datetime

# ================== ✅ ADDED ONLY ==================
import matplotlib.pyplot as plt
# ==================================================

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Models
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


# ---------------------------------------
# ⚙️ Database Configuration
# ---------------------------------------
DB_USER = "postgres"
DB_PASSWORD = "United2025"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "AutoMotor_Insurance"
DB_TABLE_NAME = "motor_insurance_data"
DB_RESULTS_TABLE = "model_training_results"


# -------------------- Engine --------------------
@st.cache_resource
def get_engine():
    try:
        return create_engine(
            f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
    except Exception as e:
        st.error(f"❌ DB Connection Failed: {e}")
        return None


# -------------------- Risk Scoring --------------------
def calculate_risk_score(vehicle_use, vehicle_age, sum_insured, driver_age):
    if str(vehicle_use).lower() == 'personal':
        vehicleuse_score = 0.2
    elif str(vehicle_use).lower() == 'commercial':
        vehicleuse_score = 1.0
    else:
        vehicleuse_score = 0.6

    if vehicle_age <= 2:
        vehicleage_score = 0.4
    elif 2 <= vehicle_age <= 5:
        vehicleage_score = 0.6
    elif 6 <= vehicle_age <= 8:
        vehicleage_score = 0.8
    else:
        vehicleage_score = 1.0

    if sum_insured <= 300000:
        suminsured_score = 0.2
    elif 300001 <= sum_insured <= 750000:
        suminsured_score = 0.4
    elif 750001 <= sum_insured <= 1500000:
        suminsured_score = 0.6
    elif 1500001 <= sum_insured <= 3000000:
        suminsured_score = 0.8
    else:
        suminsured_score = 1.0

    if driver_age < 25:
        driverage_score = 1.0
    elif 25 <= driver_age <= 35:
        driverage_score = 0.6
    elif 36 <= driver_age <= 55:
        driverage_score = 0.4
    else:
        driverage_score = 1.0

    raw_score = vehicleuse_score + vehicleage_score + suminsured_score + driverage_score

    if 1.2 <= raw_score < 1.8:
        label = "Low"
    elif 1.8 <= raw_score < 2.4:
        label = "Low to Moderate"
    elif 2.4 <= raw_score < 3.0:
        label = "Medium to High"
    else:
        label = "High"

    return raw_score, label


# -------------------- Local LLM --------------------
def is_ollama_running():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def local_llm_explanation(prompt, model_name="llama3"):
    if not is_ollama_running():
        return "⚠️ Local LLM not running. Please start Ollama using 'ollama serve'."
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt},
            timeout=180
        )
        output_text = ""
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                if '"response"' in data:
                    part = data.split('"response":"')[-1].split('"')[0]
                    output_text += part
        if not output_text:
            try:
                output_text = response.json().get('response', '') or response.text
            except Exception:
                output_text = response.text
        return output_text.strip() if output_text else "No explanation returned from local LLM."
    except Exception as e:
        return f"⚠️ Could not connect to local LLM: {e}"


# -------------------- Helpers --------------------
def model_already_trained(engine, model_name):
    try:
        query = text(f"""
            SELECT *
            FROM {DB_RESULTS_TABLE}
            WHERE model_name = :m
            ORDER BY id DESC
            LIMIT 1
        """)
        df = pd.read_sql_query(query, con=engine, params={"m": model_name})
        if not df.empty:
            return df.iloc[0].to_dict()
        return None
    except Exception:
        return None


def save_features_and_model(model_obj, X_columns, model_name):
    os.makedirs("models", exist_ok=True)
    model_file = f"models/{model_name.replace(' ','_')}.pkl"
    feat_file = f"models/{model_name.replace(' ','_')}_features.pkl"
    joblib.dump(model_obj, model_file)
    joblib.dump(X_columns, feat_file)
    return model_file, feat_file


# -------------------- Generic Training --------------------
def train_model_generic(df, model_name, model_obj):
    df.columns = df.columns.map(str)

    current_year = datetime.datetime.now().year
    df["vehicle_age"] = current_year - df.get("vehicle_make_year", current_year - 5)
    df["driver_age"] = df.get("driver_age", 30)

    df["risk_percentage"], df["risk_label"] = zip(*df.apply(lambda r:
        calculate_risk_score(
            r.get("vehicle_use", "personal"),
            r["vehicle_age"],
            r["sum_insured"],
            r["driver_age"]
        ), axis=1))

    if "policy_premium" not in df.columns:
        st.error("❌ 'policy_premium' column not found.")
        return

    target_col = "policy_premium"
    feature_cols = [c for c in df.columns if c != target_col]

    X = pd.get_dummies(df[feature_cols], drop_first=True)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    with st.spinner(f"Training {model_name}... ⏳"):
        model_obj.fit(X_train, y_train)
        y_train_pred = model_obj.predict(X_train)
        y_test_pred = model_obj.predict(X_test)

    # ================= METRICS =================
    r2 = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    mae = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    mse = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    rmse = np.sqrt(mse)
    rmse_test = np.sqrt(mse_test)

    accuracy = 100 * (1 - np.mean(np.abs((y_train - y_train_pred) / y_train)))
    testing_accuracy = 100 * (1 - np.mean(np.abs((y_test - y_test_pred) / y_test)))

    # ================= STREAMLIT METRIC CARDS =================
    st.markdown("## 📊 Model Performance")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Train R²", f"{r2:.4f}")
    col2.metric("Test R²", f"{r2_test:.4f}", delta=f"{r2_test-r2:.4f}")
    col3.metric("Train RMSE", f"{rmse:,.0f}")
    col4.metric("Test RMSE", f"{rmse_test:,.0f}")
    col5.metric("Test Accuracy", f"{testing_accuracy:.2f}%")

    # ================= TRAINING GRAPHS =================
    st.subheader("📈 Training Graphs")

    fig1, ax1 = plt.subplots()
    ax1.scatter(y_train, y_train_pred, alpha=0.4)
    ax1.plot([y_train.min(), y_train.max()],
             [y_train.min(), y_train.max()], linestyle="--")
    ax1.set_title("Training: Actual vs Predicted")
    st.pyplot(fig1)
    fig1.savefig(f"Graphs/{model_name}_train_actual_vs_pred.png", bbox_inches='tight')


    fig2, ax2 = plt.subplots()
    ax2.scatter(y_train_pred, y_train - y_train_pred, alpha=0.4)
    ax2.axhline(0, linestyle="--")
    ax2.set_title("Training: Residual Plot")
    st.pyplot(fig2)
    fig2.savefig(f"Graphs/{model_name}_train_residual.png", bbox_inches='tight')

    # ================= TESTING GRAPHS =================
    st.subheader("📉 Testing Graphs")

    fig3, ax3 = plt.subplots()
    ax3.scatter(y_test, y_test_pred, alpha=0.4)
    ax3.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], linestyle="--")
    ax3.set_title("Testing: Actual vs Predicted")
    st.pyplot(fig3)
    fig3.savefig(f"Graphs/{model_name}_test_actual_vs_predicted.png", bbox_inches='tight')

    fig4, ax4 = plt.subplots()
    ax4.scatter(y_test_pred, y_test - y_test_pred, alpha=0.4)
    ax4.axhline(0, linestyle="--")
    ax4.set_title("Testing: Residual Plot")
    st.pyplot(fig4)
    fig4.savefig(f"Graphs/{model_name}_test_residual.png", bbox_inches='tight')
    
    # ================= FEATURE IMPORTANCE =================
    if hasattr(model_obj, "feature_importances_"):
        st.subheader("🔥 Top 15 Feature Importances")

        imp_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model_obj.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(15)

        fig5, ax5 = plt.subplots()
        ax5.barh(imp_df["Feature"], imp_df["Importance"])
        ax5.invert_yaxis()
        st.pyplot(fig5)
        
    # ================= TRAINING LLM (NEW) =================
    training_prompt = f"""
    The {model_name} TRAINING results are:
    - Training R²: {r2:.4f}
    - Training MAE: {mae:,.2f}
    - Training MSE: {mse:,.2f}
    - Training RMSE: {rmse:,.2f}
    - Training Accuracy: {accuracy:.2f}%

    Analyze:
    - Model fitting behavior
    - Bias vs variance
    - Overfitting risk
    - Training reliability
    """

    #training_llm_output = local_llm_explanation(training_prompt)
    training_llm_output = local_llm_explanation(training_prompt).replace("\\n", "\n").replace("\n", " ")

    # ================= Testing LLM ANALYSIS =================
    prompt = f"""
    The {model_name} for motor insurance premium prediction produced:
    - Training R²: {r2:.4f}
    - Testing R²: {r2_test:.4f}
    - Training MAE: {mae:,.2f}
    - Testing MAE: {mae_test:,.2f}
    - Training MSE: {mse:,.2f}
    - Testing MSE: {mse_test:,.2f}
    - Training RMSE: {rmse:,.2f}
    - Testing RMSE: {rmse_test:,.2f}
    - Training Accuracy: {accuracy:.2f}%
    - Testing Accuracy: {testing_accuracy:.2f}%
    Explain what these metrics imply and provide insights on how risk factors affect prediction accuracy.
    """

    llm_output = local_llm_explanation(prompt).replace("\\n", "\n").replace("\n", " ")
    #local_llm_explanation(prompt)

    st.markdown("## 🧠 Training LLM Analysis")
    st.info(training_llm_output)

    st.markdown("## 🧠 Testing LLM Analysis")
    st.info(llm_output)

    model_file, feat_file = save_features_and_model(model_obj, X.columns.tolist(), model_name)

    # ================= SAVE TO DB =================
    try:
        engine = get_engine()
        if engine:
            results_df = pd.DataFrame([{
                "model_name": model_name,
                "r2_score": r2,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "testing_r2": r2_test,
                "testing_mae": mae_test,
                "testing_mse": mse_test,
                "testing_rmse": rmse_test,
                "accuracy": accuracy,
                "testing_accuracy": testing_accuracy,
                "llm_prompt": prompt,
                "llm_explanation": llm_output,
                # NEW FIELDS
                "training_llm_prompt": training_prompt.strip(),
                "training_llm_explanation": training_llm_output.strip()
            }])

            results_df.to_sql(DB_RESULTS_TABLE, con=engine,
                              if_exists="append", index=False)

            st.success("✅ Results saved to database!")

    except Exception as e:
        st.error(f"DB Save Error: {e}")

   #Removed
# -------------------- UI --------------------
def show():
    st.set_page_config(page_title="Unified Training Dashboard", layout="wide")
    st.title("🚀 Unified Model Training — Insurance Premiums")

    engine = get_engine()
    dataset_exists = False

    if engine is not None:
        try:
            with engine.begin() as conn:
                res = conn.execute(text(f"SELECT COUNT(*) FROM {DB_TABLE_NAME}"))
                count = res.scalar()
                dataset_exists = bool(count and count > 0)
        except Exception:
            dataset_exists = False

    if dataset_exists:
        st.warning("⚠️ Dataset already exists in the database.")
        if st.button("🗑 Delete Dataset from DB", key="delete_db_data"):
            with engine.begin() as conn:
                conn.execute(text(f"DELETE FROM {DB_TABLE_NAME}"))
            st.success("✅ Dataset deleted successfully! You can upload a new one now.")
            st.rerun()
    else:
        uploaded_file = st.file_uploader("📤 Upload dataset (CSV format)", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                df.columns = (
                    df.columns.str.strip()
                    .str.lower()
                    .str.replace(" ", "_")
                    .str.replace("-", "_")
                )
                df.columns = df.columns.map(str)
                st.success("✅ File uploaded successfully!")
                st.dataframe(df.head())

                with engine.begin() as conn:
                    df.to_sql(DB_TABLE_NAME, con=conn, if_exists="append", index=False)
                st.success("✅ Dataset saved to database!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to upload dataset: {e}")
            return

    if dataset_exists:
        try:
            df = pd.read_sql_table(DB_TABLE_NAME, con=engine)
            df.columns = df.columns.map(str)
            df.columns = (
                df.columns.str.strip()
                .str.lower()
                .str.replace(" ", "_")
                .str.replace("-", "_")
            )
            st.success("📦 Loaded dataset from database.")
            st.dataframe(df.head())

            st.write("### 🚀 Choose a Model to Train")

            cols = st.columns(5)

            models = [
                    ("XGBoost Model", XGBRegressor(
                    n_estimators=400, learning_rate=0.01, max_depth=5,
                    random_state=30, subsample=0.7, colsample_bytree=0.7,
                    objective="reg:squarederror"
                )),
                ("LightGBoost Model", LGBMRegressor(
                    n_estimators=400, learning_rate=0.01, max_depth=5,
                    subsample=0.7, colsample_bytree=0.7, random_state=30
                )),
                ("GradientBoost Model", GradientBoostingRegressor(
                     n_estimators=400,
                    learning_rate=0.01,
                    max_depth=4,       # slightly smaller than 6 to reduce overfit
                    random_state=30,
                    subsample=0.7,     # slightly lower
                    max_features=0.6
                )),
                ("CatBoost Model", CatBoostRegressor(
                     iterations=2000,   # reduced from 2500
                    learning_rate=0.01,
                    depth=5,           # reduced from 6
                    loss_function="RMSE",
                    eval_metric="RMSE",
                    random_seed=30,
                    verbose=False
                )),
                ("AdaBoost Model", AdaBoostRegressor(
                    estimator=DecisionTreeRegressor(max_depth=6),  # slightly higher depth than CatBoost
                    n_estimators=400,
                    learning_rate=0.01,
                    random_state=30
                ))
            ]
            for i, (mname, mobj) in enumerate(models):
                with cols[i]:
                    if st.button(mname, key=mname):
                        existing = model_already_trained(engine, mname)
                        if existing:
                            st.markdown(f"<h3 style='text-align:left;'>📊 {mname} Results</h3>", unsafe_allow_html=True)
                            metrics_df = pd.DataFrame({
                                "Metric": ["R²", "MAE", "MSE", "RMSE", "Accuracy"],
                                "Training": [existing['r2_score'], existing['mae'], existing['mse'],
                                             existing['rmse'], existing['accuracy']],
                                "Testing": [existing['testing_r2'], existing['testing_mae'], existing['testing_mse'],
                                            existing['testing_rmse'], existing['testing_accuracy']]
                            })
                            st.table(metrics_df)

                            st.markdown(f"""
                            <div style="background-color:#0ea5a4;padding:18px;border-radius:12px;margin-top:20px;color:#041014;">
                                <div style="font-size:20px;font-weight:600">🧠 Training LLM Explanation</div>
                                <pre style="white-space:pre-wrap;margin-top:12px; text-align:center">{existing['training_llm_explanation']}</pre>
                            </div>
                            <div style="background-color:#0ea5a4;padding:18px;border-radius:12px;margin-top:20px;color:#041014;">
                                <div style="font-size:20px;font-weight:600">🧠 Testing LLM Explanation</div>
                                <pre style="white-space:pre-wrap;margin-top:12px; text-align:center">{existing['llm_explanation']}</pre>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            train_model_generic(df.copy(), mname, mobj)

        except Exception as e:
            st.error(f"❌ Could not load dataset from DB: {e}")


if __name__ == "__main__":
    show()
