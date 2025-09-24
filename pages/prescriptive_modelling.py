import io
import json

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
# from streamlit_sortables import sortables

from utils.viz import plot_charts

def run_prescriptive_modelling():
    st.title("📋 Prescriptive Modelling")

    trained_models = st.session_state.get("trained_models", {})
    feature_importances = st.session_state.get("feature_importances", {})
    geo_unit_col = st.session_state["geo_unit_col"]
    target_col = st.session_state["target_col"]
    direction = st.session_state.get("target_direction", "Increase")
    positive_indicators = st.session_state.get("final_positive", [])
    negative_indicators = st.session_state.get("final_negative", [])
    gdf = st.session_state["gdf"]
    df =st.session_state["grouped_df"].copy()
    if not trained_models:
        st.error("⚠️ Please train models in the Home tab first.")
        st.stop()

    model_names = list(trained_models.keys())
    selected_model_name = st.selectbox("🔍 Select a trained model", model_names)
    model = trained_models[selected_model_name]
    importances = feature_importances[selected_model_name]
    selected_feats = list(importances.index)

    # Initialize the filtered feature list in session state
    if "prescriptive_filtered_feats" not in st.session_state:
        st.session_state["prescriptive_filtered_feats"] = selected_feats.copy()


    st.markdown("### 🎯 Target Indicator")

    # Initialize in session state if not present
    if "target_val" not in st.session_state:
        #st.session_state["target_val"] = float(df[target_col].mean() + 5 if direction == 'Increase' else df[target_col].mean() - 5)
        mean_val = np.round(df[target_col].mean(),2)
        if direction == 'Increase':
            st.session_state["target_val"] = float(mean_val + 5)
        else:
            st.session_state["target_val"] = float(mean_val - 5)

    # Number input that persists
    target_val = st.number_input(
        "",
        value=st.session_state["target_val"],  # Load from session state
        format="%.2f",
        label_visibility="collapsed",
        key="target_val_input"  # unique key
    )

    # Update session state whenever user changes value
    st.session_state["target_val"] = target_val


    avg_target = np.round(df[target_col].mean(),2)



    st.markdown("### 🔧 Adjust Sensitivities & View Prescriptions")

    # Add bucket selection filter
    feature_buckets_grouped = st.session_state.get("feature_buckets_grouped", {})

    # Add an "Unassigned" bucket for features not in any existing bucket
    all_grouped_feats = set(sum(feature_buckets_grouped.values(), []))
    unassigned_feats = [f for f in selected_feats if f not in all_grouped_feats]
    feature_buckets_grouped["Unassigned"] = unassigned_feats

    bucket_names = list(feature_buckets_grouped.keys())
    bucket_names.sort()

    st.markdown("---")
    st.markdown("#### 🧺 Filter features by bucket")

    def update_prescriptive_feats_on_bucket_change():
        """Updates the list of selected features based on the new bucket selection."""
        newly_filtered_feats = []
        selected_buckets = st.session_state.get("prescriptive_buckets_selector", [])
        if selected_buckets:
            for bucket in selected_buckets:
                newly_filtered_feats.extend(feature_buckets_grouped.get(bucket, []))
        else:
            newly_filtered_feats = selected_feats

        # Ensure the list of features only contains significant features from the model
        st.session_state["prescriptive_filtered_feats"] = [
            f for f in newly_filtered_feats if f in selected_feats
        ]
    st.info("Selecting no bucket is same as selecting all bucket. Hence by default all factors are available")

    st.multiselect(
        "Select one or more buckets to filter features",
        options=bucket_names,
        # default=st.session_state.get("selected_buckets", bucket_names),
        key="prescriptive_buckets_selector",
        on_change=update_prescriptive_feats_on_bucket_change,
        help="Only features in the selected buckets will be shown below."
    )
    st.session_state["selected_buckets"] = st.session_state.prescriptive_buckets_selector
    st.markdown("---")

    df_mod = df.copy()
    # For standardization-aware prescription
    scaler = st.session_state.get("scaler", None)
    feature_order = st.session_state.get("trained_feature_names", selected_feats)


    if selected_model_name == "linear":
        means = dict(zip(feature_order, scaler.mean_))
        stds = dict(zip(feature_order, scaler.scale_))

        if "final_feature_importances" in st.session_state:
            # User-edited importances
            importances_df = st.session_state["final_feature_importances"][selected_model_name]
            importances_dict = importances_df.set_index("Feature")["Importance"].to_dict()
        else:
            # Use model's coefficients
            importances_dict = dict(zip(feature_order, model.coef_))


    if "sensitivities" not in st.session_state:
        st.session_state["sensitivities"] = {}

    # --- Show Feature Summary Table (Mean & Std) ---
    #st.markdown("### 📊 Feature Summary (Current Data)")

    summary_data = []
    for feat in selected_feats:
        mean_val = df[feat].mean()
        std_val = df[feat].std()
        summary_data.append({
            "Feature": feat,
            "Mean": round(mean_val, 2),
            "Std Dev": round(std_val, 2)
        })

    summary_df = pd.DataFrame(summary_data)




    # Place this before the loop to control legend display
    show_legend = True

    # Use the filtered list of features for the main loop
    for feat in st.session_state.get("prescriptive_filtered_feats", selected_feats):
        col1, col2 = st.columns([1.4, 2.6])

        with col1:
            st.markdown(f"<div style='font-size:0.8rem; line-height:1.2rem; margin-bottom:-8px;'>{feat}</div>", unsafe_allow_html=True)
            default_sens = st.session_state["sensitivities"].get(feat, 0.5)
            sens = st.slider(
                "", 0.0, 1.0, value=default_sens, step=0.05,
                key=f"{feat}_slider", label_visibility="collapsed"
            )
            st.markdown("<div style='margin-bottom: -30px'></div>", unsafe_allow_html=True)
            st.session_state["sensitivities"][feat] = sens


        old_val = df[feat].mean()

        fi = importances.get(feat, 0.001)


        if target_val == avg_target:
            delta = pd.Series(0, index=df.index)
        else:
            delta = target_val - df[feat]


        if selected_model_name == "linear":
            std = stds.get(feat, 1.0)
            mean = means.get(feat,0)
            coef = importances_dict.get(feat, 0.001)
            adjustment = 0.0  # ✅ Default adjustment

            # ✅ Debugging info
            # Avoid division by zero or extremely small values
            if abs(coef) < 1e-6:
                st.warning(f"⚠️ Skipping {feat} due to near-zero coefficient: {coef:.2e}")
                #adjustment = 0.0
                df_mod[feat] = df[feat]  # No change
            else:


                if direction == "Increase":
                    if feat in positive_indicators:

                        adjustment = -1* abs(sens*(delta / coef))
                    elif feat in negative_indicators:

                        adjustment = abs(sens*(delta / coef))
                elif direction == "Decrease":
                    if feat in positive_indicators:

                        adjustment = abs(sens*(delta / coef))
                    elif feat in negative_indicators:

                        adjustment = -1*abs(sens*(delta / coef))


                new_val_series = df[feat] - adjustment




                min_bound, max_bound = st.session_state.feature_bounds.get(feat, (float('-inf'), float('inf')))
                new_val_series = new_val_series.clip(lower=min_bound, upper=max_bound)

                df_mod[feat] = new_val_series

        else:
            adjustment = 0.0  # ✅ Default adjustment
            if direction == "Increase":
                if feat in positive_indicators:
                    adjustment = -1* abs(sens*(delta /fi))
                elif feat in negative_indicators:
                    adjustment = abs(sens*(delta /fi))
            elif direction == "Decrease":
                if feat in positive_indicators:
                    adjustment = abs(sens*(delta / fi))
                elif feat in negative_indicators:
                    adjustment = -1*abs(sens*(delta /fi))

            new_val_series = df[feat] - adjustment
            min_bound, max_bound = st.session_state.feature_bounds.get(feat, (float('-inf'), float('inf')))
            new_val_series = new_val_series.clip(lower=min_bound, upper=max_bound)
            df_mod[feat] = new_val_series



        new_val = df_mod[feat].mean()

        with col2:
            bar = go.Figure(data=[
                go.Bar(
                    name='Old',
                    x=[old_val],
                    y=[""],
                    orientation='h',
                    marker_color='steelblue',
                    text=[f"{old_val:.2f}"],
                    textposition='auto',  # <-- auto adjusts position
                    hovertemplate='Old value: %{x:.2f}<extra></extra>',
                    showlegend=show_legend
                ),
                go.Bar(
                    name='New',
                    x=[new_val],
                    y=[""],
                    orientation='h',
                    marker_color='deepskyblue',
                    text=[f"{new_val:.2f}"],
                    textposition='auto',  # <-- auto adjusts position
                    hovertemplate='New value: %{x:.2f}<extra></extra>',
                    showlegend=show_legend
                )
            ])

            # Calculate dynamic upper limit for x-axis
            max_val = max(old_val, new_val)
            x_range = [0, max_val * 1.1]  # Add 10% padding for visibility


            bar.update_layout(
                barmode='group',
                height=120,  # Increased height for better legend and label room
                margin=dict(l=5, r=40, t=50, b=20),  # Top margin increased for legend
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.25,  # ABOVE the chart
                    xanchor="center",
                    x=0.5,
                    font=dict(size=12),
                    bgcolor="rgba(0,0,0,0)",  # transparent background
                ),
                xaxis=dict(title="", showgrid=False, showticklabels=True, range=x_range),
                yaxis=dict(showticklabels=False),
            )


            bar.update_traces(
                textfont_size=20,
                cliponaxis=False
            )

            st.plotly_chart(bar, use_container_width=True)

        show_legend = False


    st.caption("⬅️ Adjust sliders to shift factor values. Right bars show new vs old average across districts.")


    # ------------------- MAP AND BAR CHART -------------------


    if direction == "Increase":
        color_scale =  "RdYlGn"
    elif direction == "Decrease":
        color_scale = "RdYlGn_r"
    else:
        color_scale = "Viridis"


    # Predict with original and modified data
    if selected_model_name == "linear":
        #feature_order = st.session_state["trained_feature_names"]
        feature_order = st.session_state.get("trained_feature_names", selected_feats)
        scaler = st.session_state["scaler"]
        df_scaled = scaler.transform(df[feature_order])
        df_mod_scaled = scaler.transform(df_mod[feature_order])
        df["Predicted"] = model.predict(df_scaled)
        df_mod["Predicted"] = model.predict(df_mod_scaled)
    else:
        feature_order = st.session_state.get("trained_feature_names", selected_feats)
        df["Predicted"] = model.predict(df[feature_order])
        df_mod["Predicted"] = model.predict(df_mod[feature_order])

    # ✅ Apply bounds to predictions
    t_min, t_max = st.session_state.get("target_bounds", (float("-inf"), float("inf")))


    # Calculate change
    delta = df_mod["Predicted"] - df["Predicted"]
    df_mod["Predicted"] = df[target_col] + delta
    df_mod["Predicted"] = df_mod["Predicted"].clip(lower=t_min, upper=t_max)

    change = df_mod["Predicted"] - df[target_col]

    # If Decrease and change > 0, reset to actual
    if direction == 'Decrease':
        df_mod["Predicted"] = np.where(change > 0, df[target_col], df_mod["Predicted"])

    # If Increase and change < 0, reset to actual
    elif direction == 'Increase':
        df_mod["Predicted"] = np.where(change < 0, df[target_col], df_mod["Predicted"])

    df_mod["Change"] = df_mod["Predicted"] - df[target_col]




    # --- Scoreboards ---
    col1, col2, col3 = st.columns(3)

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
            <div style="font-size:15px;font-weight:600;margin-bottom:2px;">🎯 Target Value for {target_col.replace("_", " ")}</div>
            <div style="font-size:12px;color:#777;margin-bottom:6px;">set by user</div>
            <div style="font-size:24px;font-weight:700;">{target_val:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background-color:#f9ecd8;padding:12px 16px;border-radius:8px;text-align:center;">
            <div style="font-size:15px;font-weight:600;margin-bottom:2px;">Average Change in {target_col.replace("_", " ")}</div>
            <div style="font-size:12px;color:#777;margin-bottom:6px;">per District</div>
            <div style="font-size:24px;font-weight:700;">{df_mod['Change'].mean():.2f}</div>
        </div>
        """, unsafe_allow_html=True)


    change_df = df_mod[[geo_unit_col, "Change"]].copy()
    plot_charts(
        change_df=change_df,
        color_scale=color_scale,
        map_title="🗺️ District-wise Impact Visualization",
        bar_title="Impact by District"
    )

    # ------------------ DISPLAY FINAL PRESCRIPTIONS TABLE ------------------
    st.markdown("### 📋 Prescribed Feature Adjustments per District")

    display_df = df[[geo_unit_col]].copy()

    # For each selected feature, show before and after
    for feat in st.session_state.get("prescriptive_filtered_feats", selected_feats):
        display_df[f"{feat} (Current)"] = df[feat]
        display_df[f"{feat} (Prescribed)"] = df_mod[feat]

    # Target indicator before and predicted
    display_df[f"{target_col} (Current)"] = df[target_col]
    display_df[f"{target_col} (Predicted)"] = df_mod["Predicted"]
    display_df["Change in Target"] = df_mod["Change"]

    # Show in Streamlit
    st.dataframe(display_df, use_container_width=True)

    with st.expander("📤 Download Prescriptions"):
        format_choice = st.radio("Choose format:", ["CSV", "Excel"], horizontal=True)

        if format_choice == "CSV":
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name="prescriptive_modelling_output.csv",
                mime="text/csv"
            )

        elif format_choice == "Excel":
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                display_df.to_excel(writer, index=False, sheet_name="Prescriptions")
            st.download_button(
                label="⬇️ Download Excel",
                data=excel_buffer.getvalue(),
                file_name="prescriptive_modelling_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
