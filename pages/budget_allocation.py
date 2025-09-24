import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import io
from utils.budget_utils import compute_budget_allocation  
from pages.schemes_dashboard import run_schemes_dashboard
from pages.beneficiaries_dashboard import run_beneficiaries_dashboard

from utils.viz import plot_charts

def run_budget_allocation():

    st.title("Budget Allocation Dashboard")
    
    # Initialize session state key
    if "show_download_options" not in st.session_state:
        st.session_state["show_download_options"] = True  # default
        
    tab = st.radio("Select Budget View", ["Overview", "Schemes", "Beneficiaries"], horizontal=True)
    
    if tab == "Overview":

        # Use adjusted importances if available, else fallback
        feature_importances = st.session_state.get("final_feature_importances", {})
        if not feature_importances:
            feature_importances = st.session_state.get("feature_importances", {})
        feature_schemes = st.session_state.get("feature_schemes", {})  # ✅ Mapping feature → scheme
        

        pd.set_option("display.float_format", lambda x: f"{x:.17f}")  # Show up to 17 decimals


        if not feature_importances:
            st.warning("No feature importances found.")
            st.stop()


        model_names = list(feature_importances.keys())
        selected_model = st.selectbox("🧠 Select a trained model", model_names)
        df_final = st.session_state.get("grouped_df")
        feature_cols = st.session_state.get("final_selected_features", [])
        geo_unit_col = st.session_state.get("geo_unit_col", "District")  # fallback

        if df_final is None or not feature_cols:
            st.error("Required data not found in session. Please upload and configure features first.")
            st.stop()

        # Convert DataFrame of importances to dict
        df_model_importances = feature_importances.get(selected_model, pd.DataFrame())
        if isinstance(df_model_importances, pd.DataFrame):
            if df_model_importances.empty:
                st.error(f"No adjusted importances found for model `{selected_model}`.")
                st.stop()
            model_importances = dict(zip(df_model_importances["Feature"], df_model_importances["Importance"]))
        else:
            model_importances = df_model_importances

        
        
        # Add explanation and mode selection
        st.info("""
        **Budget Allocation Logic**:
        - *Current default*: Features with higher importance receive **less budget** because we assume they need fewer funds to create impact.
        - *Alternate option*: If you believe **high-impact features should get more budget**, choose the alternate mode below.
        """)
        
        
        
        # Determine default selection based on current session state
        default_option = (
            "Direct importance (high-impact gets more)"
            if st.session_state.get("allocation_mode") == "direct"
            else "Inverse importance (default)"
        )

        allocation_mode_selection = st.radio(
            "Select budget allocation logic:",
            ["Inverse importance (default)", "Direct importance (high-impact gets more)"],
            horizontal=True,
            index=0 if default_option.startswith("Inverse") else 1,
            key="allocation_mode_selection"
        )

        # Convert selection to internal value
        st.session_state["allocation_mode"] = (
            "inverse" if allocation_mode_selection.startswith("Inverse") else "direct"
        )
        

        # Map features to scheme names
        selected_importances = {
            feature_schemes.get(feature, feature): importance
            for feature, importance in model_importances.items()
            if feature in df_final.columns
        }

        total_budget = st.number_input("Enter total state budget (in ₹ Crores)", min_value=0, value=1000)

        if not selected_importances:
            st.error("No matching features found in dataset for selected model.")
            st.stop()


        
        feature_budgets, district_allocations = compute_budget_allocation(
            df_final, model_importances, total_budget, geo_unit_col, st.session_state["allocation_mode"]
        )


        # Rename feature_budgets index to scheme names
        feature_budgets.rename(index=feature_schemes, inplace=True)

        st.session_state["feature_budgets"] = feature_budgets
        st.session_state["district_allocations"] = district_allocations
        st.session_state["show_budget_results"] = True

       
        if st.session_state.get("show_budget_results"):
            st.subheader("📊 Scheme-wise Budget Allocation")

            schemes = feature_budgets.index.tolist()
            default_weights = {scheme: 1.0 for scheme in schemes}


            sensitivities = st.session_state.get("sensitivities", {})
            feature_schemes = st.session_state.get("feature_schemes", {})
            scheme_features = {v: k for k, v in feature_schemes.items()}

            # Map sensitivities from feature → scheme
            scheme_sensitivities = {
                            feature_schemes.get(feature, feature): sensitivities.get(feature, 1.0)
                            for feature in model_importances.keys()
            }
                
                
            # 1️⃣ Compute everything first
            weighted_budgets = feature_budgets.copy()

            
            for scheme in schemes:
                sensitivity = scheme_sensitivities.get(scheme, 1.0)
                weighted_budgets[scheme] *= sensitivity

            

            if weighted_budgets.sum() > 0:
                weighted_budgets = weighted_budgets / weighted_budgets.sum() * total_budget

            labels = weighted_budgets.index.tolist()
            values = weighted_budgets.values.tolist()

            # Create consistent color map for all schemes
            unique_schemes = sorted(set(labels))  # consistent order
            color_palette = px.colors.qualitative.Plotly
            colors = color_palette * (len(unique_schemes) // len(color_palette) + 1)
            color_map = dict(zip(unique_schemes, colors[:len(unique_schemes)]))

            # Map colors for this chart using the scheme names
            pie_colors = [color_map[label] for label in labels]

            # 2️⃣ Now start layout
            col1, col2 = st.columns([2, 1])

            with col2:
                with st.expander("📋 Show Scheme Legend (with Colors)"):
                    legend_html = ""
                    #for name, color in zip(labels, colors[:len(labels)]):
                    for name in labels:
                        color = color_map.get(name, "#CCCCCC")
                        legend_html += f"""
                        <div style='display: flex; align-items: center; margin-bottom: 6px;'>
                            <div style='width: 16px; height: 16px; background-color: {color}; margin-right: 10px; border-radius: 3px;'></div>
                            <div style='font-size: 14px'>{name}</div>
                        </div>
                        """
                    st.markdown(legend_html, unsafe_allow_html=True)

                #st.markdown("### 🎚️ Sensitivity Sliders")
                expand_expander = True if len(schemes) <= 4 else False
                with st.expander("Adjust Sensitivity Weights", expanded=expand_expander):

                    for scheme in schemes:
                        # Get feature name linked to scheme
                        feature = scheme_features.get(scheme)
                        if not feature:
                            continue

                        # Get current sensitivity value from the feature-level
                        current_val = sensitivities.get(feature, 1.0)

                        # Show slider
                        new_val = st.slider(
                            f"{scheme}",
                            min_value=0.0,
                            max_value=1.0,
                            value=current_val,
                            step=0.05,
                            key=f"sensitivity_slider_{scheme}"
                        )

                        # Update the feature-level sensitivity if changed
                        if new_val != current_val:
                            st.session_state["sensitivities"][feature] = new_val


            with col1:
                fig = go.Figure(
                    data=[go.Pie(
                        labels=labels,
                        values=values,
                        textinfo="percent+label",
                        #marker=dict(colors=colors[:len(labels)]),
                        marker=dict(colors=pie_colors),
                         textfont=dict(size=10, color="black", family="Arial Black"),  # BOLD TEXT
                        hole=0
                    )]
                )
                fig.update_layout(
                    title="Adjusted Budget Split by Scheme (Based on Sensitivity)",
                    showlegend=False,
                    margin=dict(l=20, r=20, t=50, b=20),
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                
        

        with st.expander("📈 View Scheme-wise Pie Chart", expanded=False):
            
            
            scheme_df = pd.DataFrame({
                "Allocated Budget (₹ Cr)": feature_budgets.values,
                "% Share": (feature_budgets / feature_budgets.sum() * 100).round(2).astype(str) + "%"
            }, index=feature_budgets.index).reset_index().rename(columns={"index": "Scheme"})
            st.write(scheme_df)

            format_choice = st.radio("Download format:", ["CSV", "Excel"], horizontal=True, key="scheme_download")
            if format_choice == "CSV":
                csv = scheme_df.to_csv(index=False)
                st.download_button(
                    label=f"⬇️ Download Scheme Budget CSV",
                    data=csv,
                    file_name="scheme_budget_breakdown.csv",
                    mime="text/csv"
                )
            else:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                    scheme_df.to_excel(writer, index=False, sheet_name="Scheme Budget")
                st.download_button(
                    label=f"⬇️ Download Scheme Budget Excel",
                    data=excel_buffer.getvalue(),
                    file_name="scheme_budget_breakdown.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                    
            

        st.subheader("📍 District-wise Total Budget Allocation")


        
        
        change_df = district_allocations[[geo_unit_col, "Total_Budget"]].rename(columns={"Total_Budget": "Change"})

        plot_charts(
            change_df=change_df,
            color_scale="RdYlGn_r",
            map_title="Geographic Distribution of Budget",
            bar_title=f"{geo_unit_col}-wise Budget Allocation"
)
        all_districts = district_allocations[[geo_unit_col, "Total_Budget"]].sort_values("Total_Budget")
        st.markdown(f"###  {geo_unit_col} Budget Allocation")
        st.dataframe(all_districts.reset_index(drop=True), use_container_width=True)

        st.markdown("#### 📥 Download")
        if st.session_state["show_download_options"]:
            format_choice = st.radio("Choose format:", ["CSV", "Excel"], horizontal=True)

            if format_choice == "CSV":
                csv = all_districts.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name="district_budget_allocation.csv",
                    mime="text/csv"
                )
            elif format_choice == "Excel":
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    all_districts.to_excel(writer, index=False, sheet_name="Budget Allocation")

                st.download_button(
                    label="⬇️ Download Excel",
                    data=excel_buffer.getvalue(),
                    file_name="district_budget_allocation.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        # ===================== DISTRICT SELECTION AND SCHEME-WISE PIE =====================
        st.subheader("🎯 District-Specific Budget Distribution")

        district_list = district_allocations[geo_unit_col].unique().tolist()
        selected_district = st.selectbox(f"Select {geo_unit_col}", district_list)

        if selected_district:
            district_row = district_allocations[district_allocations[geo_unit_col] == selected_district].iloc[0]

            district_scheme_budgets = district_row.drop([geo_unit_col, "Total_Budget"])
            district_scheme_budgets.index = [feature_schemes.get(feat, feat) for feat in district_scheme_budgets.index]

            district_scheme_names = district_scheme_budgets.index.tolist()
            district_pie_colors = [color_map.get(scheme, "#CCCCCC") for scheme in district_scheme_names]

            fig_district = go.Figure(
                data=[go.Pie(
                    labels=district_scheme_names,
                    values=district_scheme_budgets.values,
                    marker=dict(colors=district_pie_colors),
                    textinfo="percent+label",
                     textfont=dict(size=10, color="black", family="Arial Black"),  # BOLD TEXT
                    hole=0
                )]
            )
            fig_district.update_layout(
                title=f"Budget Distribution in {selected_district} (Schemes)",
                showlegend=False,
                margin=dict(l=20, r=20, t=50, b=20),
                height=500
            )
            st.plotly_chart(fig_district, use_container_width=True)

            with st.expander(f"📊 Budget Breakdown for {selected_district}", expanded=False):
                scheme_budget_df = pd.DataFrame({
                    "Scheme": district_scheme_budgets.index,
                    "Allocated Budget (₹ Cr)": district_scheme_budgets.values
                })

                total_budget_district = district_row["Total_Budget"]
                st.markdown(f"**Total Budget for {selected_district}: ₹ {total_budget_district:.2f} Cr**")
                st.dataframe(scheme_budget_df, use_container_width=True)

                format_choice = st.radio("Download format:", ["CSV", "Excel"], horizontal=True, key="district_download")
                if format_choice == "CSV":
                    csv = scheme_budget_df.to_csv(index=False)
                    st.download_button(
                        label=f"⬇️ Download {selected_district} Budget CSV",
                        data=csv,
                        file_name=f"{selected_district}_budget_breakdown.csv",
                        mime="text/csv"
                    )
                else:
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                        scheme_budget_df.to_excel(writer, index=False, sheet_name="District Budget")
                    st.download_button(
                        label=f"⬇️ Download {selected_district} Budget Excel",
                        data=excel_buffer.getvalue(),
                        file_name=f"{selected_district}_budget_breakdown.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

    elif tab == "Schemes":
        run_schemes_dashboard()

    elif tab == "Beneficiaries":
        run_beneficiaries_dashboard()