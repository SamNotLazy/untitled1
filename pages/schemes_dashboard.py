import io
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import ast

from utils.budget_utils import compute_budget_allocation
from utils.viz import plot_charts



def run_schemes_dashboard():
    st.header("📂 Budget Allocation - Schemes")

    # 1️⃣ Retrieve model importances and data
    feature_importances = (
            st.session_state.get("final_feature_importances", {})
            or st.session_state.get("feature_importances", {})
    )
    df_final = st.session_state.get("grouped_df")
    geo_unit_col = st.session_state.get("geo_unit_col", "District")
    gdf = st.session_state.get("gdf")
    GEO_COL = st.session_state.get("GEO_COL")
    feature_budgets = st.session_state.get("feature_budgets", {})
    district_allocations = st.session_state.get("district_allocations", pd.DataFrame())
    feature_schemes = st.session_state.get("feature_schemes", {})  # ✅ Fetch scheme names
    feature_buckets_grouped = st.session_state.get("feature_buckets_grouped", {})


    if not feature_importances or df_final is None or gdf is None or district_allocations.empty:
        st.error("Required data not found in session. Please ensure you have uploaded and processed the dataset.")
        return

    # 2️⃣ Select model
    #selected_model = st.radio("🧠 Select model type", options=list(feature_importances.keys()), horizontal=True)
    model_names = list(feature_importances.keys())
    selected_model = st.selectbox("🧠 Select a trained model", model_names)
    model_importances_raw = feature_importances.get(selected_model, {})

    # ✅ Convert DataFrame or stringified Series to usable dict
    if isinstance(model_importances_raw, pd.DataFrame):
        model_importances = dict(zip(model_importances_raw["Feature"], model_importances_raw["Importance"]))
    elif isinstance(model_importances_raw, str):
        try:
            model_importances = pd.Series(
                ast.literal_eval(
                    model_importances_raw.replace("dtype: float64", "")
                    .replace("dtype: float32", "")
                )
            ).to_dict()
        except Exception as e:
            st.error(f"Error parsing model importances: {e}")
            return
    else:
        model_importances = model_importances_raw

    # ✅ Recompute feature budgets and district allocations for selected model
    total_budget = st.session_state.get("total_budget", 1000)  # fallback to 1000 if not set
    allocation_mode = st.session_state.get("allocation_mode", "inverse")  # fallback
    st.info(f"Current allocation mode: **{allocation_mode}**")

    feature_budgets, district_allocations = compute_budget_allocation(
        df_final,
        model_importances,
        total_budget,
        geo_unit_col,
        mode=allocation_mode
    )



    # Update session state
    st.session_state["feature_budgets"] = feature_budgets
    st.session_state["district_allocations"] = district_allocations

    # 3️⃣ Normalize factor names
    df_final_cols = {col.strip(): col for col in df_final.columns}
    reverse_feature_schemes = {v: k for k, v in feature_schemes.items()}
    budget_keys = {reverse_feature_schemes.get(key, key).strip(): key for key in feature_budgets.keys()}
    available_factors = sorted(set(df_final_cols.keys()) & set(budget_keys.keys()))

    if not available_factors:
        st.warning("No valid features with allocated budget found.")
        return

    # Add bucket selection filter
    # Add an "Unassigned" bucket for features not in any existing bucket
    all_grouped_feats = set(sum(feature_buckets_grouped.values(), []))
    unassigned_feats = [f for f in available_factors if f not in all_grouped_feats]
    feature_buckets_grouped["Unassigned"] = unassigned_feats

    bucket_names = list(feature_buckets_grouped.keys())
    bucket_names.sort()

    st.markdown("---")
    st.markdown("#### 🧺 Filter schemes by bucket")

    def update_schemes_on_bucket_change():
        """Updates the list of schemes based on the new bucket selection."""
        newly_filtered_schemes = []
        selected_buckets = st.session_state.get("schemes_buckets_selector", [])
        if selected_buckets:
            for bucket in selected_buckets:
                # Use the feature_buckets_grouped dictionary
                newly_filtered_schemes.extend(feature_buckets_grouped.get(bucket, []))
        else:
            # If no buckets are selected, show all available factors
            newly_filtered_schemes = available_factors

        # Ensure the list of schemes only contains factors with a budget
        st.session_state["schemes_filtered_factors"] = [
            f for f in newly_filtered_schemes if f in available_factors
        ]

    st.info("Selecting no bucket is same as selecting all bucket. Hence by default all factors are available")

    st.multiselect(
        "Select one or more buckets to filter schemes",
        options=bucket_names,
        # default=st.session_state.get("selected_buckets", bucket_names),
        key="schemes_buckets_selector",
        on_change=update_schemes_on_bucket_change,
        help="Only schemes in the selected buckets will be shown below."
    )
    st.session_state["selected_buckets"] = st.session_state.schemes_buckets_selector
    st.markdown("---")

    # Use the filtered list to populate the scheme selection box
    filtered_factors = st.session_state.get("schemes_filtered_factors", available_factors)
    filtered_schemes_display = {f: feature_schemes.get(f, f) for f in filtered_factors}

    if not filtered_schemes_display:
        st.info("No schemes available in the selected bucket(s).")
        return

    selected_scheme_display = st.selectbox(
        "🎯 Select a Scheme",
        list(filtered_schemes_display.values())
    )

    selected_factor_normalized = next(f for f, s in filtered_schemes_display.items() if s == selected_scheme_display)

    selected_factor = df_final_cols[selected_factor_normalized]
    original_budget_key = budget_keys[selected_factor_normalized]
    scheme_name = feature_schemes.get(selected_factor, selected_factor)

    total_feature_budget = feature_budgets.get(original_budget_key)
    if total_feature_budget is None:
        st.warning("No budget allocated to this feature in the main overview. Please run the overview allocation first.")
        return

    st.markdown(
        f"""
        <div style="
            background-color:#fff4e5;
            padding:10px 20px;  /* Reduced vertical padding */
            border-radius:10px;
            text-align:center;
            font-family:Arial, sans-serif;
            box-shadow:0 1px 2px rgba(0,0,0,0.1);
        ">
            <h4 style="margin:0; color:#333; font-size:16px;">Total Budget Allocated to</h4>
            <h3 style="margin:5px 0; color:#000; font-size:18px;">{scheme_name}</h3>
            <p style="font-size:20px; font-weight:bold; margin:0; color:#000;">
                ₹ {total_feature_budget:.2f} Cr
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4️⃣ Distribute budget across districts proportionally
    factor_values = df_final[[geo_unit_col, selected_factor]].copy().dropna()
    total = factor_values[selected_factor].sum()
    if total == 0:
        st.error("Total of selected factor values is zero. Cannot distribute budget.")
        return

    factor_values["Allocated_Budget"] = (factor_values[selected_factor] / total) * total_feature_budget

    # 5️⃣ Visualizations
    st.subheader("📍 District-wise Scheme Budget Allocation")
    # Prepare data in the format expected by plot_charts
    change_df = factor_values[[geo_unit_col, "Allocated_Budget"]].rename(columns={"Allocated_Budget": "Change"})

    plot_charts(
        change_df=change_df,
        color_scale="RdYlGn_r",
        map_title="Geographic Distribution of Scheme Budget",
        bar_title=f"{geo_unit_col}-wise Scheme Impact"
    )


    # 6️⃣ Minimal Main Table and Download
    st.markdown("### 📄 District-wise Allocation for Selected Scheme")
    display_df = factor_values[[geo_unit_col, "Allocated_Budget"]].copy()
    display_df.columns = [geo_unit_col, f"{scheme_name} Budget"]
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

    st.markdown("#### 📥 Download")
    format_choice = st.radio("Choose format:", ["CSV", "Excel"], horizontal=True)

    if format_choice == "CSV":
        csv = display_df.to_csv(index=False)
        st.download_button("⬇️ Download CSV", data=csv, file_name=f"{scheme_name}_budget.csv", mime="text/csv")
    else:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            display_df.to_excel(writer, index=False, sheet_name="Selected Scheme Budget")
        st.download_button(
            "⬇️ Download Excel", data=excel_buffer.getvalue(),
            file_name=f"{scheme_name}_budget.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # 7️⃣ Expanded View: All District Allocations
    with st.expander("📊 Full Budget Allocation Table (All Schemes)", expanded=False):
        df_all = district_allocations.copy()
        rename_map = {col: feature_schemes.get(col, col) for col in df_all.columns if col not in [geo_unit_col, "Total_Budget"]}
        df_all.rename(columns=rename_map, inplace=True)
        columns = [geo_unit_col, "Total_Budget"] + [col for col in df_all.columns if col not in [geo_unit_col, "Total_Budget"]]
        df_all = df_all[columns]
        st.dataframe(df_all.reset_index(drop=True), use_container_width=True)

        st.markdown("#### 📥 Download Full Allocation Table")
        full_format = st.radio("Download format:", ["CSV", "Excel"], horizontal=True, key="download_format_full")

        if full_format == "CSV":
            csv_full = df_all.to_csv(index=False)
            st.download_button("⬇️ Download CSV", data=csv_full, file_name="full_district_allocations.csv", mime="text/csv")
        else:
            full_excel = io.BytesIO()
            with pd.ExcelWriter(full_excel, engine="xlsxwriter") as writer:
                df_all.to_excel(writer, index=False, sheet_name="All Scheme Budgets")
            st.download_button(
                "⬇️ Download Excel", data=full_excel.getvalue(),
                file_name="full_district_allocations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
