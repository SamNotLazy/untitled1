import io
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
# from streamlit_sortables import sortables

from utils.viz import plot_charts


def run_differential_impact_analysis():
    st.title("📉 Differential Impact Analysis")

    trained_models = st.session_state.get("trained_models", {})

    # --- Get final importances if available ---
    if "final_feature_importances" in st.session_state:
        feature_importances = {
            model: df.set_index("Feature")["Importance"]
            for model, df in st.session_state["final_feature_importances"].items()
        }
    else:
        feature_importances = st.session_state.get("feature_importances", {})

    scaler = st.session_state.get("scaler", None)

    if not trained_models:
        st.error("🚫 No models have been trained yet. Please train models on the Home page first.")
        st.stop()

    # --- Model selector ---
    model_names = list(trained_models.keys())
    selected_model_name = st.selectbox("🔍 Select a trained model", model_names)
    model = trained_models[selected_model_name]
    sig_feats = list(feature_importances[selected_model_name].index)

    # --- Load dataset ---
    df = st.session_state["grouped_df"].copy()
    geo_unit_col = st.session_state["geo_unit_col"]
    target_col = st.session_state["target_col"]
    gdf = st.session_state["gdf"]
    GEO_COL = st.session_state["GEO_COL"]

    if not sig_feats:
        st.error("🚫 No selected features available for analysis.")
        st.stop()

    # --- Feature selection and sliders ---
    st.markdown("### 🛠️ Select Features and Define % Interventions")

    # New: Add bucket selection filter
    feature_buckets_grouped = st.session_state.get("feature_buckets_grouped", {})

    # Add an "Unassigned" bucket for features not in any existing bucket
    all_grouped_feats = set(sum(feature_buckets_grouped.values(), []))
    unassigned_feats = [f for f in sig_feats if f not in all_grouped_feats]
    feature_buckets_grouped["Unassigned"] = unassigned_feats

    bucket_names = list(feature_buckets_grouped.keys())
    bucket_names.sort()

    st.markdown("---")
    st.markdown("#### 🧺 Filter features by bucket")

    def update_intervene_feats_on_bucket_change():
        """Updates the list of selected features based on the new bucket selection."""
        # Get the newly filtered features from the current bucket selection
        newly_filtered_feats = []
        selected_buckets = st.session_state.get("selected_buckets_selector", [])
        if selected_buckets:
            for bucket in selected_buckets:
                newly_filtered_feats.extend(feature_buckets_grouped.get(bucket, []))
        else:
            # If no buckets are selected, fall back to all significant features
            newly_filtered_feats = sig_feats

        # Ensure the list of features to intervene on only contains features from the new selection
        st.session_state["intervene_feats"] = [
            f for f in st.session_state.get("intervene_feats", [])
            if f in newly_filtered_feats
        ]
    st.info("Selecting no bucket is same as selecting all bucket. Hence by default all factors are available")
    selected_buckets = st.multiselect(
        "Select one or more buckets to filter features",
        options=bucket_names,
        # default=st.session_state.get("selected_buckets", bucket_names),
        key="selected_buckets_selector",
        on_change=update_intervene_feats_on_bucket_change,
        help="Only features in the selected buckets will be shown below."
    )
    st.session_state["selected_buckets"] = selected_buckets
    st.markdown("---")

    # Filter features based on selected buckets
    filtered_feats = []
    if selected_buckets:
        for bucket in selected_buckets:
            filtered_feats.extend(feature_buckets_grouped.get(bucket, []))
    else:
        filtered_feats = sig_feats

    # Ensure filtered list contains only significant features from the model
    filtered_feats = [f for f in filtered_feats if f in sig_feats]

    if "intervene_feats" not in st.session_state:
        st.session_state["intervene_feats"] = []

    intervene_feats = st.multiselect(
        "📊 Choose Features to Intervene On",
        filtered_feats,
        default=[f for f in st.session_state.get("intervene_feats", []) if f in filtered_feats],
        key="intervene_feats_selector"
    )
    st.session_state["intervene_feats"] = intervene_feats

    default_sensitivities = st.session_state.get("sensitivities", {})
    pct_changes = {}
    sensitivities = {}

    if "pct_changes" not in st.session_state:
        st.session_state["pct_changes"] = {}

    for feat in intervene_feats:
        st.markdown(f"#### 🔧 `{feat}`")
        col1, col2 = st.columns(2)

        with col1:
            default_pct = st.session_state["pct_changes"].get(feat, 0.0) * 100
            pct = st.slider(
                "📉 % Change",
                min_value=-100.0,
                max_value=100.0,
                value=default_pct,
                step=1.0,
                key=f"{feat}_pct_change"
            ) / 100.0
            st.session_state["pct_changes"][feat] = pct

        with col2:
            default_sens = default_sensitivities.get(feat, 0.5)
            sens = st.slider(
                "🎯 Sensitivity",
                min_value=0.0,
                max_value=1.0,
                value=default_sens,
                step=0.05,
                key=f"{feat}_sensitivity"
            )

        pct_changes[feat] = pct
        sensitivities[feat] = sens
        st.markdown("---")

    st.session_state["sensitivities"] = sensitivities

    # --- Apply interventions ---
    df_mod = df.copy()
    for feat in pct_changes:
        pct = pct_changes[feat]
        sens = sensitivities[feat]
        df_mod[feat] = df_mod[feat] * (1 + sens * pct)

    feature_order = st.session_state.get("trained_feature_names", sig_feats)

    # --- Model predictions ---
    if selected_model_name == "linear":
        df_scaled = scaler.transform(df[feature_order])
        df_mod_scaled = scaler.transform(df_mod[feature_order])

        if "final_feature_importances" in st.session_state:
            user_importances = st.session_state["final_feature_importances"][selected_model_name]
            coef = user_importances.set_index("Feature").reindex(feature_order)["Importance"].values
            df["Predicted"] = df_scaled @ coef
            df_mod["Predicted"] = df_mod_scaled @ coef
        else:
            df["Predicted"] = model.predict(df_scaled)
            df_mod["Predicted"] = model.predict(df_mod_scaled)
    else:
        df["Predicted"] = model.predict(df[feature_order])
        df_mod["Predicted"] = model.predict(df_mod[feature_order])

    df_mod["Change"] = df_mod["Predicted"] - df["Predicted"]

    # --- Apply direction masking ---
    direction = st.session_state.get("target_direction", "Increase")
    positive_indicators = st.session_state.get("final_positive", [])
    negative_indicators = st.session_state.get("final_negative", [])

    for feat in pct_changes:
        pct = pct_changes[feat]
        if direction == "Increase":
            if feat in positive_indicators:
                df_mod["Change"] = df_mod["Change"].mask((pct > 0) & (df_mod["Change"] < 0), 0)
                df_mod["Change"] = df_mod["Change"].mask((pct < 0) & (df_mod["Change"] > 0), 0)
            elif feat in negative_indicators:
                df_mod["Change"] = df_mod["Change"].mask((pct > 0) & (df_mod["Change"] > 0), 0)
                df_mod["Change"] = df_mod["Change"].mask((pct < 0) & (df_mod["Change"] < 0), 0)
        elif direction == "Decrease":
            if feat in positive_indicators:
                df_mod["Change"] = df_mod["Change"].mask((pct > 0) & (df_mod["Change"] < 0), 0)
                df_mod["Change"] = df_mod["Change"].mask((pct < 0) & (df_mod["Change"] > 0), 0)
            elif feat in negative_indicators:
                df_mod["Change"] = df_mod["Change"].mask((pct > 0) & (df_mod["Change"] > 0), 0)
                df_mod["Change"] = df_mod["Change"].mask((pct < 0) & (df_mod["Change"] < 0), 0)

    # --- Color scale ---
    if direction == "Increase":
        color_scale = "RdYlGn"
    elif direction == "Decrease":
        color_scale = "RdYlGn_r"
    else:
        color_scale = "Viridis"

    # --- Scorecards ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background-color:#f9ecd8;padding:12px 16px;border-radius:8px;text-align:center;">
            <div style="font-size:15px;font-weight:600;margin-bottom:2px;">Current {target_col.replace("_", " ")}</div>
            <div style="font-size:12px;color:#777;margin-bottom:6px;">per District</div>
            <div style="font-size:24px;font-weight:700;">{df[target_col].mean():.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background-color:#f9ecd8;padding:12px 16px;border-radius:8px;text-align:center;">
            <div style="font-size:15px;font-weight:600;margin-bottom:2px;">Average Change in {target_col.replace("_", " ")}</div>
            <div style="font-size:12px;color:#777;margin-bottom:6px;">per District</div>
            <div style="font-size:24px;font-weight:700;">{df_mod['Change'].mean():.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Plot charts ---
    plot_charts(
        change_df=df_mod[[geo_unit_col, "Change"]],
        color_scale=color_scale,
        map_title="Geographic Distribution of Change",
        bar_title=f"{geo_unit_col}-wise Impact (Sorted)"
    )
    
    
    
    # =========================================
    # Marginal change per feature (absolute Δ)
    # =========================================
    st.subheader("Marginal Change by Feature (others at 0%)")

    # Helper: same prediction path as the main section
    def _predict_df(Xdf):
        if selected_model_name == "linear":
            if scaler is None:
                st.error("Scaler missing for linear model.")
                st.stop()
            Xs = scaler.transform(Xdf[feature_order])
            if "final_feature_importances" in st.session_state:
                uimp = st.session_state["final_feature_importances"][selected_model_name]
                coef = uimp.set_index("Feature").reindex(feature_order)["Importance"].values
                return Xs @ coef  # intercept cancels for deltas
            else:
                return model.predict(Xs)
        else:
            return model.predict(Xdf[feature_order])

    # Pull direction metadata
    direction = st.session_state.get("target_direction", "Increase")
    pos = set(st.session_state.get("final_positive", []))
    neg = set(st.session_state.get("final_negative", []))

    # Baseline predictions (reuse exactly what main used)
    # df["Predicted"] already computed above in your code
    pred_base = df["Predicted"].to_numpy()

    marginal_rows = []

    for feat in intervene_feats:
        # Start from original df (no other interventions)
        df_only = df.copy()

        pct = float(pct_changes.get(feat, 0.0))
        sens = float(sensitivities.get(feat, 0.0))

        # Apply ONLY this feature's intervention
        df_only[feat] = df_only[feat] * (1 + sens * pct)

        # Predict with same pipeline
        pred_only = _predict_df(df_only)

        # Raw change for this feature alone
        change = pd.Series(pred_only - pred_base, index=df.index)

        # Apply the SAME direction masking rules, but only for this feature
        if direction == "Increase":
            if feat in pos:
                change = change.mask((pct > 0) & (change < 0), 0)
                change = change.mask((pct < 0) & (change > 0), 0)
            elif feat in neg:
                change = change.mask((pct > 0) & (change > 0), 0)
                change = change.mask((pct < 0) & (change < 0), 0)
        elif direction == "Decrease":
            if feat in pos:
                change = change.mask((pct > 0) & (change > 0), 0)
                change = change.mask((pct < 0) & (change < 0), 0)
            elif feat in neg:
                change = change.mask((pct > 0) & (change < 0), 0)
                change = change.mask((pct < 0) & (change > 0), 0)

        marginal_rows.append({"Feature": feat, "Avg Δ": float(change.mean())})

    if marginal_rows:
        marginal_df = pd.DataFrame(marginal_rows).sort_values("Avg Δ", ascending=True)

        # If exactly one feature is intervened, this will match the scorecard mean
        # (df_mod['Change'].mean()) by construction.
        # Optional: tiny sanity caption
        if len(marginal_df) == 1:
            st.caption(
                f"Sanity check: marginal Avg Δ = {marginal_df['Avg Δ'].iloc[0]:.4f} "
                f"| scorecard Avg Δ = {df_mod['Change'].mean():.4f}"
            )

        
        # same range as the map (zero-centered)
        cmin = float(df_mod["Change"].min())
        cmax = float(df_mod["Change"].max())
        M = max(abs(cmin), abs(cmax))
        cmin, cmax = -M, M   # symmetric around 0

        fig_marginal = px.bar(
            marginal_df,
            y="Feature",
            x="Avg Δ",
            text="Avg Δ",
            color="Avg Δ",                               # <- color by the value
            color_continuous_scale=color_scale,          # <- same as map ("RdYlGn" or "RdYlGn_r")
            range_color=(cmin, cmax),                    # <- same domain as map
            color_continuous_midpoint=0,                 # <- diverging around zero
            title="Marginal Average Change per Feature"
        )

        # (optional) hide the colorbar if you don't want it
        fig_marginal.update_coloraxes(showscale=False)

        # spacing/thickness
        fig_marginal.update_traces(width=0.85, texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
        fig_marginal.update_layout(bargap=0.10, bargroupgap=0, height=360,
                                xaxis_title=f"Average change in {target_col}", yaxis_title="",
                                margin=dict(t=60, r=20, l=20, b=40), showlegend=False)


        # Optional: smaller tick/label fonts for long feature names
        fig_marginal.update_yaxes(automargin=True, tickfont=dict(size=11))
        fig_marginal.update_xaxes(tickfont=dict(size=11))

        

        fig_marginal.update_yaxes(automargin=True, categoryorder="array",
                                  categoryarray=marginal_df["Feature"].tolist())
        #st.plotly_chart(fig_marginal, use_container_width=True)
        st.plotly_chart(fig_marginal, use_container_width=True,
                key=f"marginal_feat_{target_col}")
    else:
        st.info("Select at least one feature to see marginal changes.")

    # --- Final data table ---
    st.markdown("##### 📋 Modified Predictions per District")
    display_df = df[[geo_unit_col]].copy()
    for feat in intervene_feats:
        display_df[f"{feat} (Before)"] = df[feat]
        display_df[f"{feat} (After)"] = df_mod[feat]

    display_df[target_col] = df[target_col]
    display_df["Predicted"] = df[target_col] + df_mod["Change"]
    display_df["Change"] = df_mod["Change"]
    st.dataframe(display_df, use_container_width=True)

    with st.expander("📤 Select Download Format"):
        format_choice = st.radio("Choose format:", ["CSV", "Excel"], horizontal=True)
        if format_choice == "CSV":
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name="differential_impact_analysis.csv",
                mime="text/csv"
            )
        elif format_choice == "Excel":
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                display_df.to_excel(writer, index=False, sheet_name="Intervention Impact")
            st.download_button(
                label="⬇️ Download Excel",
                data=excel_buffer.getvalue(),
                file_name="differential_impact_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
