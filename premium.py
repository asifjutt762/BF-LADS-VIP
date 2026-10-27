import streamlit as st
import pandas as pd
import datetime
from sqlalchemy import create_engine, text
import requests
import joblib

# ============================================
# Database Configuration
# ============================================
DB_USER = "postgres"
DB_PASSWORD = "United2025"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "AutoMotor_Insurance"
DB_TABLE_NAME1 = "premium_results"
DB_VEHICLE_TABLE = "motor_vehicle"

# ============================================
# Local LLM (Ollama) - unchanged
# ============================================
def is_ollama_running():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200
    except:
        return False

def local_llm_explanation(prompt, model_name="llama3"):
    if not is_ollama_running():
        return "⚠️ Local LLM not running. Start Ollama using: ollama serve"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt},
            timeout=300
        )

        output = ""
        for line in response.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if '"response":"' in decoded:
                    part = decoded.split('"response":"')[-1].split('"')[0]
                    output += part

        return output.strip()
    except Exception as e:
        return f"⚠️ LLM Connection Error: {e}"

# ============================================
# Database Engine
# ============================================
@st.cache_resource
def get_engine():
    try:
        eng = create_engine(
            f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
        return eng
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

# ============================================
# Helper
# ============================================
def normalize_key(s: str):
    if s is None:
        return ""
    return s.strip().lower()

def pretty_display(s: str):
    if s is None or s == "":
        return ""
    return s.strip().title()

# ============================================
# Main App
# ============================================
def show():
    st.title("🚗 Motor Insurance Premium Prediction")

    engine = get_engine()
    if not engine:
        st.stop()

    # Model from session_state
    model = st.session_state.get("loaded_model")
    selected_model = st.session_state.get("selected_model_name")
    feature_cols = st.session_state.get("model_features")

    if model is None and "model_check_initialized" in st.session_state:
        st.warning("⚠️ No model loaded. Please go to 'Model Validation' first.")
        return

    st.session_state["model_check_initialized"] = True

    # Risk profile
    risk_profile_selected = st.session_state.get("risk_profile_value")
    if not risk_profile_selected:
        st.warning("⚠️ Please complete the Risk Profile step first.")
        return

    current_year = datetime.datetime.now().year

    # ================================
    # FETCH VEHICLE MAKES
    # ================================
    try:
        with engine.connect() as conn:
            sql = text(f"""
                SELECT DISTINCT lower(trim(vehicle_make)) AS make_norm
                FROM {DB_VEHICLE_TABLE}
                WHERE vehicle_make IS NOT NULL AND trim(vehicle_make) <> ''
                ORDER BY make_norm
            """)
            df_makes = pd.read_sql(sql, conn)

        make_norm_list = df_makes["make_norm"].dropna().unique().tolist()
        make_display_list = [pretty_display(m) for m in make_norm_list]
        make_display_to_norm = {pretty_display(m): m for m in make_norm_list}

    except Exception as e:
        st.error(f"❌ Failed loading vehicle makes: {e}")
        make_display_list = []
        make_display_to_norm = {}

    st.subheader("Enter Vehicle Details")

    if len(make_display_list) == 0:
        st.info("No vehicle makes found in database.")
        return

    vehicle_make_display = st.selectbox("Vehicle Make", options=make_display_list)

    # ================================
    # FETCH MODELS
    # ================================
    model_list_display = []
    model_display_to_norm = {}

    try:
        selected_make_norm = make_display_to_norm.get(vehicle_make_display, normalize_key(vehicle_make_display))
        with engine.connect() as conn:
            sql_models = text("""SELECT DISTINCT lower(trim(vehicle_model)) AS model_norm
                                 FROM motor_vehicle
                                 WHERE lower(trim(vehicle_make)) = :make_norm
                                 AND vehicle_model IS NOT NULL AND trim(vehicle_model) <> ''
                                 ORDER BY model_norm""")
            df_models = pd.read_sql(sql_models, conn, params={"make_norm": selected_make_norm})

        model_norm_list = df_models["model_norm"].dropna().unique().tolist()
        model_list_display = [pretty_display(m) for m in model_norm_list]
        model_display_to_norm = {pretty_display(m): m for m in model_norm_list}

    except Exception as e:
        st.error(f"❌ Failed loading vehicle models for {vehicle_make_display}: {e}")

    if len(model_list_display) == 0:
        vehicle_model_display = st.selectbox("Vehicle Model", options=["-- No models found --"])
    else:
        vehicle_model_display = st.selectbox("Vehicle Model", options=model_list_display)

    # SESSION VALUES
    session_sum_insured = st.session_state.get("risk_sum_insured")
    session_vehicle_year = st.session_state.get("risk_vehicle_year")

    # Vehicle Year
    if session_vehicle_year is not None:
        st.number_input(
            "Vehicle Make Year (from Risk Profile)",
            min_value=1980,
            max_value=current_year,
            value=int(session_vehicle_year),
            disabled=True
        )
        vehicle_make_year = int(session_vehicle_year)
    else:
        vehicle_make_year = st.number_input(
            "Vehicle Make Year",
            min_value=1980, max_value=current_year, value=current_year - 3
        )

    # Sum Insured
    if session_sum_insured is not None:
        st.number_input(
            "Sum Insured (from Risk Profile)",
            min_value=1, value=int(session_sum_insured), disabled=True
        )
        sum_insured = float(session_sum_insured)
    else:
        sum_insured = st.number_input("Sum Insured", min_value=1, value=500000)

    st.text_input("Risk Profile", value=risk_profile_selected, disabled=True)

    if not st.button("Predict Premium"):
        return

    # ================================
    # LOAD MODEL & FEATURES
    # ================================
    try:
        if not model or not feature_cols:
            model_filename = f"models/{selected_model.replace(' ','_')}.pkl"
            features_filename = f"models/{selected_model.replace(' ','_')}_features.pkl"
            model = joblib.load(model_filename)
            feature_cols = joblib.load(features_filename)
            st.session_state["loaded_model"] = model
            st.session_state["model_features"] = feature_cols
    except Exception as e:
        st.error(f"❌ Failed to load model/features: {e}")
        return

    # ================================
    # BUILD FEATURE ROW
    # ================================
    make_norm = normalize_key(vehicle_make_display)
    model_norm = normalize_key(vehicle_model_display)

    def build_feature_row(sum_insured, vehicle_make_year, vehicle_age, make_norm, model_norm, feature_cols):
        row = pd.DataFrame([[0]*len(feature_cols)], columns=feature_cols)

        # Numeric columns
        for col, val in [("sum_insured", sum_insured),
                         ("vehicle_make_year", vehicle_make_year),
                         ("vehicle_age", vehicle_age)]:
            if col in row.columns:
                row.at[0, col] = val

        # One-hot categorical columns
        vmk_col = f"vehicle_make_{make_norm}".replace(" ", "_").lower()
        vmd_col = f"vehicle_model_{model_norm}".replace(" ", "_").lower()

        if vmk_col in row.columns:
            row.at[0, vmk_col] = 1
        if vmd_col in row.columns:
            row.at[0, vmd_col] = 1

        return row

    vehicle_age = current_year - int(vehicle_make_year)
    X_row = build_feature_row(sum_insured, vehicle_make_year, vehicle_age,
                              make_norm, model_norm, feature_cols)

    # ================================
    # PREDICT
    # ================================
    try:
        raw_prediction = float(model.predict(X_row)[0])
        predicted_premium = max(raw_prediction, 0)
        predicted_rate = predicted_premium / sum_insured * 100
        source = f"🤖 Based on {selected_model}" if selected_model else "🤖 Model"
    except Exception as e:
        st.error("Prediction failed at model.predict(X_row)")
        st.error(f"Actual error: {e}")
        predicted_rate = 1.50
        predicted_premium = (predicted_rate / 100) * sum_insured
        source = "📊 Model Error — Used Default Rate"

    # Adjust by risk profile
    risk_map = {"Low": 0.05, "Low to Moderate": 0.075, "Moderate to High": 0.10, "High": 0.15}
    final_premium = predicted_premium * (1 + risk_map.get(risk_profile_selected, 0.10))
    final_rate = final_premium / sum_insured * 100

    # ================================
    # DISPLAY RESULTS
    # ================================
    st.subheader("🔍 Prediction Results")
    st.success(f"Predicted Premium: {predicted_premium:,.2f} | Rate: {predicted_rate:.2f}%")
    st.warning(f"Final Premium: {final_premium:,.2f} | Rate: {final_rate:.2f}%")

    # ================================
    # LLM Explanation (optional)
    # ================================
    st.subheader("🧠 Local AI Explanation")
    prompt = (
        f"Explain premium logic. Final premium: {final_premium:.2f}, "
        f"rate: {final_rate:.2f}%, risk profile: {risk_profile_selected}, "
        f"vehicle age: {vehicle_age}, sum insured: {sum_insured}."
    )
    llm_explanation = local_llm_explanation(prompt).replace("\\n", "\n").replace("\n", " ")  # local_llm_explanation(prompt) if needed
    st.write(llm_explanation)

    # ================================
    # SAVE RESULTS TO DB
    # ================================
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name='premium_results'
                        AND column_name='llm_explanation'
                    ) THEN
                        ALTER TABLE premium_results ADD COLUMN llm_explanation TEXT;
                    END IF;
                END$$;
            """))

            conn.execute(
                text(f"""
                    INSERT INTO {DB_TABLE_NAME1}
                    (vehicle_make, vehicle_model, vehicle_make_year, sum_insured, vehicle_age,
                     risk_profile, predicted_premium, predicted_rate, final_premium, final_rate,
                     prediction_source, created_at, llm_prompt, llm_explanation)
                    VALUES
                    (:m, :mod, :year, :sum, :age, :risk,
                     :pprem, :prate, :fprem, :frate, :src, :ts, :prompt, :llmexp)
                """),
                {
                    "m": pretty_display(vehicle_make_display),
                    "mod": pretty_display(vehicle_model_display),
                    "year": int(vehicle_make_year),
                    "sum": float(sum_insured),
                    "age": int(vehicle_age),
                    "risk": risk_profile_selected,
                    "pprem": float(predicted_premium),
                    "prate": float(predicted_rate),
                    "fprem": float(final_premium),
                    "frate": float(final_rate),
                    "src": source,
                    "ts": datetime.datetime.now(),
                    "prompt": prompt,
                    "llmexp": llm_explanation
                }
            )
        st.success("✅ Prediction & Full LLM Prompt Saved!")
    except Exception as e:
        st.error(f"❌ Failed to save: {e}")

    # ================================
    # VIEW STORED RESULTS
    # ================================
    st.subheader("📑 View Stored Results")
    if st.button("View Stored Results"):
        try:
            df_premium = pd.read_sql("SELECT * FROM premium_results", engine)
            df_premium = df_premium.loc[:, ~df_premium.columns.duplicated()]
            st.dataframe(df_premium, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Could not load results: {e}")

# Run Page
show()