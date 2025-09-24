import json
import os
import joblib
import geopandas as gpd
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from streamlit.errors import StreamlitValueAssignmentNotAllowedError
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pages.home import load_geo
from utils.data_prep import check_linear_regression_assumptions


pd.set_option("display.float_format", lambda x: f"{x:.17f}")  # Show up to 17 decimals

# --- STREAMLIT UI AND MODEL SETUP ---


# ----------------- HELPER FUNCTIONS -----------------
def train_model(model_name, X_train, y_train, final_features, params=None):
    feature_importances = None
    scaler = None
    if model_name == "rf":
        model = RandomForestRegressor(random_state=42, **(params or {}))
        model.fit(X_train, y_train)
        feature_importances = pd.Series(model.feature_importances_, index=final_features)
    elif model_name == "xgb":
        model = XGBRegressor(random_state=42, verbosity=0, **(params or {}))
        model.fit(X_train, y_train)
        feature_importances = pd.Series(model.feature_importances_, index=final_features)

    elif model_name == "linear":
        if st.session_state.get("use_standardized", True):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
        model = LinearRegression()
        model.fit(X_train, y_train)
        feature_importances = pd.Series(np.abs(model.coef_), index=final_features)
    return model, feature_importances, scaler


def evaluate_models(trained_models, selected_models, X_test, y_test, final_features, scaler=None, title_suffix=""):
    metrics_data = []
    for model_name in selected_models:
        model = trained_models[model_name]
        if model_name == "linear" and scaler:
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        metrics_data.append({
            "Model": model_name.upper(),
            "R² Score": round(r2, 4),
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "MAPE (%)": round(mape * 100, 4)
        })

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df)

    #fig = make_subplots(rows=2, cols=2, subplot_titles=["R² Score", "MAE", "MSE", "MAPE (%)"])
    fig = make_subplots(rows=1, cols=4, subplot_titles=["R² Score", "MAE", "MSE", "MAPE (%)"])

    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]
    metrics = ["R² Score", "MAE", "MSE", "MAPE (%)"]


    for i, metric in enumerate(metrics):
        row = 1
        col = i + 1
        fig.add_trace(
            go.Bar(
                x=metrics_df["Model"],
                y=metrics_df[metric],
                name=metric,
                marker_color=colors[i],
                showlegend=False
            ),
            row=row, col=col
        )


    fig.update_layout(
        height=400, width=1100,
        title_text=f"📊 Evaluation Metrics by Model {title_suffix}",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)


def train_all_models(selected_models, X_train, y_train, final_features, custom_params=None):
    trained_models = {}
    feature_importances = {}
    scaler = None
    for model_name in selected_models:
        params = (custom_params or {}).get(model_name, {})
        model, fi, s = train_model(model_name, X_train, y_train, final_features, params)
        trained_models[model_name] = model
        feature_importances[model_name] = fi
        if s:
            scaler = s
    return trained_models, feature_importances, scaler






def run_intervention_setup():

    st.title("Intervention Dashboard Overview")

    # # --- ADDED BUTTONS FOR SAVE/LOAD ---
    # col1, col2 = st.columns(2)
    # with col1:
    #     if st.button("💾 Save Session"):
    #         save_session_state()
    # with col2:
    #     if st.button("📂 Load Session"):
    #         load_session_state()
    # # --- END ADDED BUTTONS ---

    st.markdown('<div id="page_top"></div>', unsafe_allow_html=True)
    st.markdown("""
        Welcome to the **Intervention Dashboard**. Use the links in the sidebar to explore:
        - **📉 Differential Impact Analysis**
        - **🔧 Prescriptive Modelling**
        - **💰 Budget Allocation**
    """)

    st.markdown("### ✏️ Define Min & Max Constraints for Each Feature")
    st.info("ℹ️ Set the allowed value range for each selected feature")

    if "final_selected_features" in st.session_state and st.session_state["final_selected_features"]:
        if "feature_bounds" not in st.session_state:
            st.session_state.feature_bounds = {}
        if "feature_schemes" not in st.session_state:
            st.session_state.feature_schemes = {}
        if "sensitivities" not in st.session_state:
            st.session_state.sensitivities = {}

        feature_bounds = {}
        feature_schemes = {}
        sensitivities = {}

        df_final = st.session_state["uploaded_df"].copy()
        target_col = st.session_state.get("target_col", None)

        # --- Target Variable Bounds ---
        if target_col:
            st.markdown(f"#### 🎯 Target Variable: `{target_col}`")
            target_min = float(df_final[target_col].min())
            target_max = float(df_final[target_col].max())
            col1, col2 = st.columns(2)
            with col1:
                t_min = st.number_input(
                    f"🔽 Minimum for `{target_col}`", value=target_min, step=1.0, format="%.2f", key="target_min"
                )
            with col2:
                t_max = st.number_input(
                    f"🔼 Maximum for `{target_col}`", value=target_max, step=1.0, format="%.2f", key="target_max"
                )

        st.session_state.target_bounds = (t_min, t_max)

        # --- Feature Bounds & Sensitivity ---
        for feature in st.session_state["final_selected_features"]:
            st.markdown(f"#### 🧾 Feature: `{feature}`")

            # --- Scheme Name ---
            default_scheme = st.session_state.feature_schemes.get(feature, feature)
            scheme_name = st.text_input(
                f"🏷️ Enter Scheme Name for `{feature}`",
                value=default_scheme,
                key=f"{feature}_scheme"
            )
            feature_schemes[feature] = scheme_name



            # --- Min/Max Bounds ---

            feature_min = float(df_final[feature].min())
            feature_max = float(df_final[feature].max())
            col1, col2 = st.columns(2)
            with col1:

                min_val = st.number_input(
                    f"🔽 Minimum for `{feature}`", value=feature_min, step=1.0, format="%.2f", key=f"{feature}_min"
                )
            with col2:

                max_val = st.number_input(
                    f"🔼 Maximum for `{feature}`", value=feature_max, step=1.0, format="%.2f", key=f"{feature}_max"
                )



            feature_bounds[feature] = (min_val, max_val)

            # --- Sensitivity Slider ---
            default_sens = st.session_state.sensitivities.get(feature, 0.5)
            sens = st.slider(
                f"🎚️ Sensitivity for `{feature}`", min_value=0.0, max_value=1.0,
                value=default_sens, step=0.05, key=f"{feature}_sensitivity"
            )
            sensitivities[feature] = sens

        # --- Save all updated values to session state ---
        st.session_state.feature_bounds = feature_bounds
        st.session_state.feature_schemes = feature_schemes
        st.session_state.sensitivities = sensitivities



    # --- MODEL SELECTION ---
    st.markdown("### 🧠 Select Models to Train")
    selected_models = []


    if st.checkbox("Linear Regression", key="linear_chk"):
        selected_models.append("linear")
        lr_scale_option = st.radio(
            "📏 Choose feature scaling for Linear Regression",
            options=["Standardized", "Unstandardized"],
            index=0,  # Default to "Standardized"
            horizontal=True
        )
        st.session_state["use_standardized"] = (lr_scale_option == "Standardized")


    if st.checkbox("Random Forest"):
        selected_models.append("rf")
    if st.checkbox("XGBoost"):
        selected_models.append("xgb")
    st.session_state["selected_models"] = selected_models

    # --- TRAIN BUTTON ---
    if selected_models:
        if st.button("🚀 Train Selected Models"):
            st.session_state["trigger_train"] = True

    # --- MAIN TRAINING LOGIC ---
    if (
            "uploaded_df" in st.session_state
            and "mapping" in st.session_state
            and "geo_unit_col" in st.session_state
            and "final_selected_features" in st.session_state
            and "target_col" in st.session_state
            and st.session_state["final_selected_features"]
            and st.session_state.get("trigger_train")
    ):
        df_final = st.session_state["uploaded_df"].copy()
        mapping = st.session_state["mapping"]
        geo_unit_col = st.session_state["geo_unit_col"]
        target_col = st.session_state["target_col"]
        final_features = st.session_state["final_selected_features"]

        for feat in final_features:
            min_val, max_val = st.session_state["feature_bounds"].get(feat, (None, None))
            if min_val is not None:
                df_final = df_final[df_final[feat] >= min_val]
            if max_val is not None:
                df_final = df_final[df_final[feat] <= max_val]

        df_final[geo_unit_col] = df_final[geo_unit_col].map(mapping)
        df_final = df_final[[geo_unit_col, *final_features, target_col]]
        # df_final = clean_df(df_final, target_col)
        st.session_state["df_final"] = df_final

        X = df_final[final_features]
        y = df_final[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.session_state["X_train"] = X_train
        st.session_state["y_train"] = y_train
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test

        # --- HYPERPARAMETER TUNING ---
        if "best_params" not in st.session_state:
            st.session_state["best_params"] = {}
        best_params = st.session_state["best_params"]
        status_placeholder = st.empty()
        with st.spinner("🔍 Performing hyperparameter tuning and training for selected models..."):
            trained_models, feature_importances, scaler = train_all_models(selected_models, X_train, y_train, final_features, st.session_state["best_params"])
            if scaler:
                st.session_state["scaler"] = scaler


        # X_train_scaled = st.session_state.get("scaler").transform(X_train) if "scaler" in st.session_state else X_train
        # pass_flag, assumption_report = check_linear_regression_assumptions(X_train_scaled, y_train, final_features)
        if st.session_state.get("use_standardized", True) and "scaler" in st.session_state:
            X_for_check = st.session_state["scaler"].transform(X_train)
        else:
            X_for_check = X_train
        pass_flag, assumption_report = check_linear_regression_assumptions(X_for_check, y_train, final_features)

        st.markdown("### 🔍 Linear Regression Assumptions Check")
        st.caption(f"Using {'standardized' if st.session_state['use_standardized'] else 'original'} features for Linear Regression.")


        # Display assumption results
        for assumption, result in assumption_report.items():
            if result["status"] == "pass":
                st.success(f"✅ {assumption}: Satisfied (Test Statistic: {result.get('statistic', 'N/A')})")
            else:
                st.error(f"❌ {assumption}: Failed (Test Statistic: {result.get('statistic', 'N/A')})")

        # Display summary
        if all(result["status"] == "pass" for result in assumption_report.values()):
            st.info("All linear regression assumptions appear to be satisfied.")
        else:
            st.warning("Some assumptions are not satisfied. Review the above results.")


        st.session_state["trained_models"] = trained_models
        st.session_state["feature_importances"] = feature_importances
        st.session_state["best_params"] = best_params
        #    st.session_state["custom_params"] = best_params.copy()
        del st.session_state["trigger_train"]
        # st.session_state["trigger_train"] = False
        st.success("✅ Training completed successfully!")
        # ✅ Trigger scroll & keep expander open only once after training
        st.session_state["trigger_scroll_to_top"] = True
        st.session_state["show_hparam_expander"] = True



        # --- METRICS DISPLAY AFTER TRAINING ---
        st.markdown("### 📈 Model Evaluation on Test Set")
        evaluate_models(trained_models, selected_models, X_test, y_test, final_features, scaler=st.session_state.get("scaler"), title_suffix="(Split View)")






    # --- HYPERPARAMETER TUNING UI ---
    if "best_params" in st.session_state:
        st.markdown("### 🎛️ Hyperparameter Tuning (Optional)")
        if "custom_params" not in st.session_state:
            st.session_state["custom_params"] = {k: v.copy() for k, v in st.session_state["best_params"].items()}
        # ✅ Scroll once after training
        if st.session_state.get("trigger_scroll_to_top", False):
            st.markdown("""
                <script>
                    const top = document.getElementById("page_top");
                    if (top) {
                        top.scrollIntoView({behavior: "smooth"});
                    }
                </script>
            """, unsafe_allow_html=True)
            st.session_state["trigger_scroll_to_top"] = False  # 🧼 Immediately reset the flag

        # ✅ THEN render the expander (not before)
        expand_expander = st.session_state.get("show_hparam_expander", False)

        # 🧼 Reset BEFORE rendering the expander so next rerun stays collapsed
        st.session_state["show_hparam_expander"] = False

        #with st.expander("⚙️ Adjust Hyperparameters", expanded=expand_expander):
        with st.expander("⚙️ Adjust Hyperparameters"):


            for model_name in selected_models:
                st.subheader(f"{model_name.upper()} Parameters")
                current_params = st.session_state["custom_params"].get(model_name, {})
                new_params = {}

                if model_name == "rf":
                    new_params["n_estimators"] = st.slider("n_estimators", 10, 300, current_params.get("n_estimators", 100), key=f"rf_n")
                    new_params["max_depth"] = st.slider("max_depth", 1, 30, current_params.get("max_depth", 10), key=f"rf_d")
                    new_params["min_samples_split"] = st.slider("min_samples_split", 2, 20, current_params.get("min_samples_split", 2), key=f"rf_s")
                    new_params["min_samples_leaf"] = st.slider("min_samples_leaf", 1, 10, current_params.get("min_samples_leaf", 1), key=f"rf_l")

                elif model_name == "xgb":
                    new_params["n_estimators"] = st.slider("n_estimators", 10, 300, current_params.get("n_estimators", 100), key=f"xgb_n")
                    new_params["max_depth"] = st.slider("max_depth", 1, 20, current_params.get("max_depth", 6), key=f"xgb_d")
                    new_params["learning_rate"] = st.slider("learning_rate", 0.01, 0.5, float(current_params.get("learning_rate", 0.1)), step=0.01, key=f"xgb_lr")
                    new_params["subsample"] = st.slider("subsample", 0.1, 1.0, float(current_params.get("subsample", 0.8)), step=0.05, key=f"xgb_ss")

                elif model_name == "linear":
                    st.markdown("No tunable hyperparameters for Linear Regression.")
                    continue

                st.session_state["custom_params"][model_name] = new_params


            if st.button("🔁 Retrain Models"):
                X_train = st.session_state["X_train"]
                y_train = st.session_state["y_train"]
                X_test = st.session_state["X_test"]
                y_test = st.session_state["y_test"]
                final_features = st.session_state["final_selected_features"]

                trained_models, feature_importances, scaler = train_all_models(selected_models, X_train, y_train, final_features, st.session_state["custom_params"])
                if scaler:
                    st.session_state["scaler"] = scaler

                st.session_state["trained_models"] = trained_models
                st.session_state["feature_importances"] = feature_importances

                st.markdown("### 📈 Updated Evaluation After Retraining")
                evaluate_models(trained_models, selected_models, X_test, y_test, final_features, scaler=st.session_state.get("scaler"), title_suffix="(Retrained)")
                st.success("✅ Retraining completed and metrics updated.")




                # --- FINAL TRAINING ---
    if "trained_models" in st.session_state and st.session_state["trained_models"]:
        if st.button("🎉 Finalize Models"):

            X_full = pd.concat([st.session_state["X_train"], st.session_state["X_test"]])
            y_full = pd.concat([st.session_state["y_train"], st.session_state["y_test"]])
            final_features = st.session_state["final_selected_features"]

            final_models = {}
            with st.spinner("🚀 Training models on full dataset..."):
                for model_name in selected_models:
                    params = st.session_state["custom_params"].get(model_name, {})
                    model, _, scaler = train_model(model_name, X_full, y_full, final_features, params)
                    if scaler:
                        st.session_state["scaler"] = scaler
                    final_models[model_name] = model


            st.session_state["final_models"] = final_models
            st.success("✅ Final models trained on entire dataset.")

        # --- FEATURE IMPORTANCE ADJUSTMENT ---
        if "final_models" in st.session_state:
            st.markdown("### 🛠️ Adjust Final Feature Importances (Optional)")

            if "final_feature_importances" not in st.session_state:
                st.session_state["final_feature_importances"] = {}

            # Use finalized factors (from correlation check) if available
            positive_factors = st.session_state.get("final_positive", [])
            negative_factors = st.session_state.get("final_negative", [])

            for model_name, model in st.session_state["final_models"].items():
                st.subheader(f"📌 Feature Importances for `{model_name.upper()}`")
                final_features = st.session_state["final_selected_features"]

                # Get initial importances
                if model_name == "linear":
                    importances = model.coef_
                elif model_name in ["rf", "xgb"]:
                    importances = model.feature_importances_
                else:
                    importances = np.zeros(len(final_features))

                # Check if adjusted importances already exist in session
                if model_name not in st.session_state["final_feature_importances"]:
                    st.session_state["final_feature_importances"][model_name] = pd.DataFrame({
                        "Feature": final_features,
                        "Importance": importances
                    })

                df_importance = st.session_state["final_feature_importances"][model_name]

                # --- Linear model: check sign mismatches ---
                mismatches = []
                if model_name == "linear":
                    for feat, coef in zip(final_features, importances):
                        if feat in positive_factors and coef < 0:
                            mismatches.append((feat, coef))
                        if feat in negative_factors and coef > 0:
                            mismatches.append((feat, coef))

                    if mismatches:
                        st.warning(f"⚠️ {len(mismatches)} mismatches detected. Example: {mismatches[0][0]} ({mismatches[0][1]:.17f})")

                        # ✅ Persistent button state
                        btn_key = f"adjust_btn_{model_name}"
                        if btn_key not in st.session_state:
                            st.session_state[btn_key] = False

                        if st.button(f"⚠️ Adjust Coefficients for Multicollinearity ({model_name.upper()})", key=f"btn_{model_name}"):
                            st.session_state[btn_key] = True

                        if st.session_state[btn_key]:
                            # Adjust and persist
                            adjusted_importances = []
                            for feat, coef in zip(final_features, importances):
                                abs_coef = abs(coef)
                                if feat in negative_factors:
                                    adjusted_importances.append(-abs_coef)
                                else:
                                    adjusted_importances.append(abs_coef)

                            df_importance["Importance"] = adjusted_importances
                            model.coef_ = np.array(adjusted_importances)
                            st.session_state["final_models"][model_name] = model
                            st.session_state["final_feature_importances"][model_name] = df_importance

                            st.success("✅ Coefficients adjusted and model updated successfully.")
                            st.write("#### Original vs Adjusted")

                            st.dataframe(pd.DataFrame({
                                "Feature": final_features,
                                "Original": [f"{x:.17f}" for x in importances],
                                "Adjusted": [f"{x:.17f}" for x in adjusted_importances]
                            }))
                            st.info("Predictions will now use adjusted coefficients.")

                # --- Tree models: enforce direction ---
                if model_name in ["rf", "xgb"]:
                    adjusted_importances = []
                    for feat, imp in zip(final_features, importances):
                        abs_imp = abs(imp)
                        if feat in negative_factors:
                            adjusted_importances.append(-abs_imp)
                        else:
                            adjusted_importances.append(abs_imp)
                    df_importance["Importance"] = adjusted_importances
                    total = df_importance["Importance"].abs().sum()
                    if total > 0:
                        df_importance["Importance"] /= total
                    st.session_state["final_feature_importances"][model_name] = df_importance
                    st.info("Tree models normalized with enforced direction for visualization.")



                # Editable UI (always show latest)
                st.markdown("#### ✏️ Edit Importances Manually")
                # Show rounded copy in editor
                editable_df = df_importance.copy()
                editable_df["Importance"] = editable_df["Importance"].apply(lambda x: f"{x:.17f}")
                # Display with text-based editing (preserves full precision)
                edited_df = st.data_editor(
                    editable_df,
                    key=f"editor_{model_name}",
                    num_rows="dynamic",
                    column_config={
                        "Importance": st.column_config.TextColumn(
                            "Importance (editable, full precision)",
                            help="Enter a number like -0.00018366162461970134"
                        )
                    }
                )

                # Try converting back to float
                try:
                    edited_df["Importance"] = edited_df["Importance"].astype(float)
                except ValueError:
                    st.error("Some Importance values are not valid floats.")

                st.session_state["final_feature_importances"][model_name] = edited_df
