import io
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import ast

from utils.budget_utils import compute_budget_allocation
from utils.viz import plot_charts



def run_beneficiaries_dashboard():
    st.header("👥 Budget Allocation - Beneficiaries")


    # 1️⃣ Retrieve session data
    feature_importances = st.session_state.get("final_feature_importances", {})
    if not feature_importances:
        feature_importances = st.session_state.get("feature_importances", {})

    df_final = st.session_state.get("grouped_df")
    geo_unit_col = st.session_state.get("geo_unit_col", "District")
    gdf = st.session_state.get("gdf")
    GEO_COL = st.session_state.get("GEO_COL")
    feature_schemes = st.session_state.get("feature_schemes", {})
    selected_features = st.session_state.get("final_selected_features", [])
    feature_buckets_grouped = st.session_state.get("feature_buckets_grouped", {})


    if not feature_importances or df_final is None or gdf is None:
        st.error("Required data not found in session. Please ensure you've run the Overview tab first.")
        return

    # 2️⃣ Select model
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


    # 3️⃣ Recompute or reuse feature budgets and district allocations
    total_budget = st.session_state.get("total_budget", 1000)  # fallback to 1000 Cr
    allocation_mode = st.session_state.get("allocation_mode", "inverse")  # default
    st.info(f"Current allocation mode: **{allocation_mode}**")



    # Always recompute when model changes
    feature_budgets, district_allocations = compute_budget_allocation(
        df_final,
        model_importances,
        total_budget,
        geo_unit_col,
        mode=allocation_mode
    )


    # Optionally update session state
    st.session_state["feature_budgets"] = feature_budgets
    st.session_state["district_allocations"] = district_allocations



    # 4️⃣ Feature selection with budget normalization
    df_final_cols = {col.strip(): col for col in df_final.columns}

    # Clean budget_keys mapping
    budget_keys = {}
    for fb_key in feature_budgets.keys():
        for df_col in df_final.columns:
            if df_col.strip() == fb_key.strip():
                budget_keys[df_col.strip()] = fb_key
                break


    available_factors = sorted(set(df_final_cols.keys()) & set(budget_keys.keys()))
    if not available_factors:
        st.warning("No valid features with allocated budget found.")
        return

    # Initialize the filtered feature list in session state
    if "beneficiaries_filtered_factors" not in st.session_state:
        st.session_state["beneficiaries_filtered_factors"] = available_factors.copy()


    # 5️⃣ Initialize scheme unit costs if not present
    if "scheme_unit_costs" not in st.session_state:
        st.session_state["scheme_unit_costs"] = {}
    for factor in available_factors:
        if factor not in st.session_state["scheme_unit_costs"]:
            scheme_name = feature_schemes.get(factor, factor)
            scheme_budget = feature_budgets.get(scheme_name, 0)
            st.session_state["scheme_unit_costs"][factor] = max((scheme_budget * 1e7) / 1000, 1.0)


    # Add bucket selection filter
    # Add an "Unassigned" bucket for features not in any existing bucket
    all_grouped_feats = set(sum(feature_buckets_grouped.values(), []))
    unassigned_feats = [f for f in available_factors if f not in all_grouped_feats]
    feature_buckets_grouped["Unassigned"] = unassigned_feats

    bucket_names = list(feature_buckets_grouped.keys())
    bucket_names.sort()

    st.markdown("---")
    st.markdown("#### 🧺 Filter schemes by bucket")

    def update_beneficiaries_on_bucket_change():
        """Updates the list of schemes based on the new bucket selection."""
        newly_filtered_schemes = []
        selected_buckets = st.session_state.get("beneficiaries_buckets_selector", [])
        if selected_buckets:
            for bucket in selected_buckets:
                newly_filtered_schemes.extend(feature_buckets_grouped.get(bucket, []))
        else:
            newly_filtered_schemes = available_factors

        st.session_state["beneficiaries_filtered_factors"] = [
            f for f in newly_filtered_schemes if f in available_factors
        ]
    st.info("Selecting no bucket is same as selecting all bucket. Hence by default all factors are available")

    st.multiselect(
        "Select one or more buckets to filter schemes",
        options=bucket_names,
        # default=st.session_state.get("selected_buckets", bucket_names),
        key="beneficiaries_buckets_selector",
        on_change=update_beneficiaries_on_bucket_change,
        help="Only schemes in the selected buckets will be shown below."
    )
    st.session_state["selected_buckets"] = st.session_state.beneficiaries_buckets_selector
    st.markdown("---")


    # 5.1 Adjust unit costs for all schemes
    with st.expander("⚙️ Adjust Unit Costs for All Schemes", expanded=False):
        st.info("Modify the unit cost (₹ per beneficiary) for each scheme. Default values are set for 1000 beneficiaries.")
        # Use the filtered list for unit cost adjustments
        for factor in st.session_state.get("beneficiaries_filtered_factors", available_factors):
            scheme_name = feature_schemes.get(factor, factor)
            current_cost = st.session_state["scheme_unit_costs"].get(
                factor, (feature_budgets.get(scheme_name, 0) * 1e7) / 1000
            )
            current_cost = max(current_cost, 1.0)
            new_cost = st.number_input(
                f"Unit Cost for '{scheme_name}' (₹ per beneficiary)",
                min_value=1.0,
                value=float(current_cost),
                step=100.0,
                key=f"uc_all_{factor}"
            )
            st.session_state["scheme_unit_costs"][factor] = new_cost

    # 6️⃣ Scheme selection
    # Use the filtered list for the scheme selection box
    filtered_factors = st.session_state.get("beneficiaries_filtered_factors", available_factors)
    scheme_display_names = {factor: feature_schemes.get(factor, factor) for factor in filtered_factors}
    if not scheme_display_names:
        st.info("No schemes available in the selected bucket(s).")
        return

    selected_scheme_display = st.selectbox(
        "🎯 Select a Scheme",
        list(scheme_display_names.values())
    )

    selected_factor_normalized = next(f for f, s in scheme_display_names.items() if s == selected_scheme_display)
    selected_factor = df_final_cols[selected_factor_normalized]
    original_budget_key = budget_keys[selected_factor_normalized]
    scheme_name = feature_schemes.get(selected_factor, selected_factor)
    total_feature_budget = feature_budgets.get(original_budget_key, 0)


    # 7️⃣ Unit cost and beneficiaries calculation
    current_unit_cost = st.session_state["scheme_unit_costs"].get(
        selected_factor, (total_feature_budget * 1e7) / 1000
    )
    current_unit_cost = max(current_unit_cost, 1.0)
    unit_cost = st.number_input(
        f"💰 Enter unit cost for '{scheme_name}' (in ₹ per beneficiary)",
        min_value=1.0,
        value=float(current_unit_cost),
        step=100.0,
        key=f"uc_{selected_factor}"
    )
    st.session_state["scheme_unit_costs"][selected_factor] = unit_cost
    total_beneficiaries = (total_feature_budget * 1e7) / unit_cost

    # 8️⃣ Title Box
    st.markdown(
        f"""
        <div style="
            background-color:#fff4e5;
            padding:10px 20px;
            border-radius:10px;
            text-align:center;
            font-family:Arial, sans-serif;
            box-shadow:0 1px 2px rgba(0,0,0,0.1);
        ">
            <h4 style="margin:0; color:#333; font-size:16px;">Total Budget Allocated to</h4>
            <h3 style="margin:5px 0; color:#000; font-size:18px;">{scheme_name}</h3>
            <p style="font-size:18px; font-weight:bold; margin:0; color:#000;">
                ₹ {total_feature_budget:.2f} Cr → Estimated Beneficiaries: {round(total_beneficiaries):,}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 9️⃣ District-wise beneficiaries
    if selected_factor not in district_allocations.columns:
        st.error(f"No budget data found for scheme '{scheme_name}'. Run Schemes Dashboard first.")
        return

    factor_values = district_allocations[[geo_unit_col, selected_factor]].copy()
    factor_values.rename(columns={selected_factor: "Allocated_Budget"}, inplace=True)

    total_allocated = factor_values["Allocated_Budget"].sum()
    if total_allocated == 0:
        st.error("Total allocated budget for this scheme is zero. Cannot distribute beneficiaries.")
        return

    factor_values["Beneficiaries"] = (factor_values["Allocated_Budget"] * 1e7) / unit_cost

    # 🔟 Visualization
    st.subheader("📍 District-wise Beneficiary Distribution")
    # Prepare input DataFrame
    change_df = factor_values[[geo_unit_col, "Beneficiaries"]].copy()
    change_df.rename(columns={"Beneficiaries": "Change"}, inplace=True)

    # Use reusable chart function
    plot_charts(
        change_df=change_df,
        color_scale="RdYlGn_r",
        map_title="Geographic Distribution of Beneficiaries",
        bar_title=f"{geo_unit_col}-wise Beneficiaries",
    )


    # 1️⃣1️⃣ Table: Beneficiaries for selected scheme
    st.markdown("### 📄 District-wise Beneficiaries for Selected Scheme")
    display_df = factor_values[[geo_unit_col, "Beneficiaries"]].copy()
    display_df["Beneficiaries"] = display_df["Beneficiaries"].round().astype(int)
    display_df.columns = [geo_unit_col, f"{scheme_name} Beneficiaries"]
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

    st.markdown("#### 📥 Download")
    format_choice = st.radio("Choose format:", ["CSV", "Excel"], horizontal=True, key="benef_format_1")

    if format_choice == "CSV":
        st.download_button("⬇️ Download CSV", data=display_df.to_csv(index=False),
                           file_name=f"{scheme_name}_beneficiaries.csv", mime="text/csv")
    else:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            display_df.to_excel(writer, index=False, sheet_name="Beneficiaries")
        st.download_button("⬇️ Download Excel", data=excel_buffer.getvalue(),
                           file_name=f"{scheme_name}_beneficiaries.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # 1️⃣2️⃣ Table: All scheme beneficiaries
    with st.expander("📊 Full Beneficiaries Table (All Schemes)", expanded=False):
        st.info("This shows beneficiaries for all schemes across districts.")
        full_df = district_allocations.copy()

        if not full_df.empty:
            for col in full_df.columns:
                if col not in [geo_unit_col, "Total_Budget"]:
                    scheme_cost = st.session_state["scheme_unit_costs"].get(col, 1.0)
                    full_df[col] = ((full_df[col] * 1e7) / scheme_cost).round().astype(int)

            beneficiary_cols = [c for c in full_df.columns if c not in [geo_unit_col, "Total_Budget"]]
            full_df["Total_Beneficiaries"] = full_df[beneficiary_cols].sum(axis=1)

            rename_map = {col: feature_schemes.get(col, col) for col in full_df.columns
                          if col not in [geo_unit_col, "Total_Beneficiaries", "Total_Budget"]}
            full_df.rename(columns=rename_map, inplace=True)

            cols = [geo_unit_col, "Total_Beneficiaries"] + \
                   [c for c in full_df.columns if c not in [geo_unit_col, "Total_Budget", "Total_Beneficiaries"]]
            display_full = full_df[cols]

            st.dataframe(display_full.reset_index(drop=True), use_container_width=True)

            full_format = st.radio("Download format:", ["CSV", "Excel"], horizontal=True, key="benef_format_2")
            if full_format == "CSV":
                st.download_button("⬇️ Download CSV", data=display_full.to_csv(index=False),
                                   file_name="all_scheme_beneficiaries.csv", mime="text/csv")
            else:
                excel_buf = io.BytesIO()
                with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
                    display_full.to_excel(writer, index=False, sheet_name="All Beneficiaries")
                st.download_button("⬇️ Download Excel", data=excel_buf.getvalue(),
                                   file_name="all_scheme_beneficiaries.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
