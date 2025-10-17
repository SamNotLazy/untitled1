import io
import json

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from functools import partial
from pages.prescription_detail_view import run_detail_prescriptions_view
# from streamlit_sortables import sortables
from streamlit_js_eval import streamlit_js_eval
st.set_page_config(layout='wide')

from utils.viz import plot_charts
st.set_page_config(
    initial_sidebar_state="collapsed"
)
def change_sensitivity(feat):
    value=st.session_state[f"{feat}_presription_sensitivity_slider"]
    st.session_state["sensitivities"][feat]=value
    st.rerun()
def extract_district_names(selection_data):
    """
    Extracts a list of district names from the 'points' array
    within the Streamlit/Plotly selection data structure.

    Args:
        selection_data (dict): The dictionary containing the selection details.

    Returns:
        list: A list of strings, where each string is a district name.
    """
    district_list = []

    # 1. Check if the top-level 'selection' key exists
    if 'selection' in selection_data and isinstance(selection_data['selection'], dict):
        selection = selection_data['selection']

        # 2. Check if the 'points' array exists
        if 'points' in selection and isinstance(selection['points'], list):

            # 3. Iterate through each point object
            for point in selection['points']:

                # 4. Check if 'properties' dictionary exists and contains 'district'
                if 'properties' in point and 'district' in point['properties']:

                    # 5. Extract the district name and add it to the list
                    district_name = point['properties']['district']
                    district_list.append(district_name)

    return district_list

def create_map_figure(plot_df, color_scale, highlight_districts_list=None):
    """
    Creates and returns the Plotly choropleth map figure,
    with an option to highlight specific districts.

    :param plot_df: GeoPandas DataFrame containing data and geometry.
    :param color_scale: Plotly color scale for the main choropleth layer.
    :param highlight_districts_list: List of district names (based on geo_key) to highlight.
    """
    if highlight_districts_list is None:
        highlight_districts_list = []

    geo_key = st.session_state["geo_unit_col"]
    map_col = "Change"
    GEO_COL = st.session_state["GEO_COL"]
    gdf = st.session_state["gdf"] # Note: Assuming gdf is the full GeoDataFrame if plot_df is a subset

    page_width = streamlit_js_eval(js_expressions='window.innerWidth', key='WIDTH',  want_output = True,)
    # st.info(page_width)
    # print(page_width)
    if page_width==None:
        zoom=4.2
    else:

        # st.write(page_width)
        zoom=4.2


    # --- Choropleth Map (Base Layer) ---
    map_fig = px.choropleth_mapbox(
        plot_df,
        geojson=plot_df.__geo_interface__,
        locations=plot_df.index, # Use DataFrame index for selection tracking
        color=map_col,
        hover_name=geo_key, # District name as the main hover label
        hover_data={
            map_col: ':.2f', # Show Change value formatted
        },
        mapbox_style="white-bg",
        opacity=0.7,
        color_continuous_scale=color_scale,
        labels={map_col: 'Change'},
        center={"lat": float(plot_df.geometry.unary_union.centroid.y), "lon":float(plot_df.geometry.unary_union.centroid.x)},
        zoom=zoom,
    )
    print(highlight_districts_list)
    # print(plot_df)
    # --- Highlighting Trace ---
    if highlight_districts_list:
        # Filter the DataFrame to include only the districts to highlight
        highlight_df = plot_df[plot_df[geo_key].isin(highlight_districts_list)]
        print(highlight_df)
        if not highlight_df.empty:
            # Create a new trace with transparent fill but thick, colored boundaries
            map_fig.add_trace(go.Choroplethmapbox(
                geojson=highlight_df.__geo_interface__,
                locations=highlight_df.index,
                # Use a dummy color value (z) and a transparent color scale
                z=[0] * len(highlight_df),
                colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
                marker_line_width=3, # Thicker line for emphasis
                marker_line_color='red', # Highlight color (e.g., bright red or blue)
                hoverinfo='text',
                text=highlight_df[geo_key], # Only show the name on hover
                name="Highlighted Districts",
                showlegend=False,
                showscale=False
            ))


    # Conditional Label Trace
    if 'geometry' in plot_df.columns and len(plot_df) > 0:
        # Add labels for districts using centroid
        map_fig.add_trace(go.Scattermapbox(
            lon=[g.centroid.x for g in plot_df.geometry],
            lat=[g.centroid.y for g in plot_df.geometry],
            mode='text',
            text=plot_df.apply(lambda row: f"{row[geo_key]}<br>{row[map_col]:.2f}", axis=1),
            textfont=dict(size=8, color='black'),
            hoverinfo='none',
            showlegend=False
        ))


    # Remove Index from Hover Tooltip using hovertemplate
    # Note: Using the first trace (index 0) which is the main choropleth
    map_fig.data[0].hovertemplate = f'<b>%{{hovertext}}</b><br>Change: %{{z:.2f}}<extra></extra>'


    # Force map to fit bounds with dynamic padding
    # map_fig.update_mapboxes(
    #     bounds={
    #         "west": float(minx),
    #         "east": float(maxx),
    #         "south": float(miny),
    #         "north": float(maxy)
    #     }
    # )

    map_fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=380,
        coloraxis_colorbar=dict(
            # Set orientation to horizontal
            orientation="h",
            # Set the length of the horizontal colorbar (1.0 means full width of the plot area)
            len=0.7,
            # Position above the plot (y=1.0 is the top edge of the map)
            y=1.02,
            # Center the colorbar horizontally
            x=0.5,
            xanchor="center",
            # Remove the colorbar title if necessary, or adjust its position

        )
    )

    return map_fig

def create_bar_figure(plot_df, color_scale, bar_title, selected_districts):
    """Creates and returns the Plotly horizontal bar chart figure."""
    geo_key = st.session_state["geo_unit_col"]
    bar_col = "Change"

    # Filter the data for the bar chart
    if selected_districts:
        # Filter the data to only include selected districts
        bar_df = plot_df[plot_df[geo_key].isin(selected_districts)].copy()
        current_bar_title = f"{bar_title} (Selected: {len(selected_districts)})"
    else:
        # If nothing is selected (or default was empty), show all districts
        bar_df = plot_df.copy()
        current_bar_title = bar_title

    # Dynamic height calculation based on number of bars
    num_bars = len(bar_df)
    bar_height = max(400, num_bars * 25)

    # --- Bar Chart Figure Generation ---
    bar_fig = px.bar(
        bar_df.sort_values(bar_col, ascending=True), # Use the filtered/full bar_df
        x=bar_col, y=geo_key, orientation='h',
        title=current_bar_title, # Use the dynamic title
        labels={geo_key: 'District', bar_col: 'Change'},
        color=bar_col,
        color_continuous_scale=color_scale,
        height=bar_height,
        text=bar_df[bar_col].round(2), # Use bar_df for text
        hover_data={
            geo_key: False, # Name is the Y-axis label, no need for redundant hover label
            bar_col: ':.2f'
        }
    )
    bar_fig.update_traces(
        texttemplate="%{text}",
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=10, color="black"),
        cliponaxis=False
    )
    bar_fig.update_coloraxes(showscale=False)

    bar_fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis=dict(tickfont=dict(size=10)),
        # height=380
    )
    return bar_fig

def create_feature_comparison_charts(df, df_mod, selected_districts):
    """
    Creates a set of dual bar charts comparing current vs. prescribed values for selected features
    across the currently selected districts, or the overall average if none are selected.

    MODIFIED: Reduced chart height, title font size, and margins to be more compact.
    """
    filtered_feats = st.session_state.get("prescriptive_filtered_feats", [])
    geo_key = st.session_state["geo_unit_col"]

    if not filtered_feats:
        st.caption("No features selected for prescriptive modelling.")
        return

    # --- MODIFIED LOGIC: Use all data if no districts are selected ---
    if selected_districts:
        # If districts are selected, filter the data to the selected districts
        df_filtered = df[df[geo_key].isin(selected_districts)]
        df_mod_filtered = df_mod[df_mod[geo_key].isin(selected_districts)]
        chart_title_suffix = f" (Avg over {len(selected_districts)} districts)"
    else:
        # If NO districts are selected, use the entire dataset (overall average)
        df_filtered = df
        df_mod_filtered = df_mod
        chart_title_suffix = " (Overall Average)"


    # Create a container for scrollability
    # Reduced container height slightly to reduce gap
    with st.container(height=380):
        # Determine if we need to show the legend (only once)
        show_legend_once = False

        for feat in filtered_feats:
            # Calculate mean values for the currently selected subset of districts
            old_val = df_filtered[feat].mean()
            # Use the value calculated in the core logic block after clipping to bounds
            new_val = df_mod_filtered[feat].mean()

            # --- Start UI for Title and Popover (for synchronised slider) ---
            # Use columns to place the title and the popover trigger side-by-side
            # Increased title column width slightly to reduce text wrapping on small screens
            title_col, popover_col = st.columns([0.80, 0.20])

            with title_col:
                # MODIFICATION: Reduced font size from 14px to 12px for compactness
                st.markdown(f"<div style='margin-top: 0px;font-size: 12px;'>**{feat.replace('_', ' ')}** {chart_title_suffix}</div>", unsafe_allow_html=True)

            with popover_col:
                # Popover trigger button
                with st.popover("⚙️", width=True):
                    st.markdown(f"**Adjust Sensitivity for {feat.replace('_', ' ')}**")
                    default_sens = st.session_state["sensitivities"][feat]

                # The key f"{feat}_slider" ensures this slider is SYNCHRONOUS
                    # with the main slider below in run_prescriptive_modelling.

                    value=st.slider(
                        "Sensitivity", 0.0, 1.0, value=default_sens, step=0.05,
                        key=f"{feat}_presription_sensitivity_slider",
                        label_visibility="visible",
                        on_change=partial(change_sensitivity,feat)
                    )






            # --- End UI for Title and Popover ---

            # Create the dual bar chart for this specific feature
            bar = go.Figure(data=[
                go.Bar(
                    name='Current',
                    x=[old_val],
                    y=[""],
                    orientation='h',
                    marker_color='steelblue',
                    text=[f"{old_val:.2f}"],
                    textposition='auto',
                    hovertemplate='Current value: %{x:.2f}<extra></extra>',
                    showlegend=show_legend_once
                ),
                go.Bar(
                    name='Prescribed',
                    x=[new_val],
                    y=[""],
                    orientation='h',
                    marker_color='deepskyblue',
                    text=[f"{new_val:.2f}"],
                    textposition='auto',
                    hovertemplate='Prescribed value: %{x:.2f}<extra></extra>',
                    showlegend=show_legend_once
                )
            ])

            # Calculate dynamic upper limit for x-axis
            max_val = max(old_val, new_val)
            x_range = [0, max_val * 1.2] if max_val > 0 else [0, 1]

            # MODIFICATION: Reduced height from 150 to 80 for compactness
            bar.update_layout(
                # Title is now handled outside the figure for popover placement
                title=" ",
                barmode='group',
                height=80,
                # MODIFICATION: Reduced bottom margin from 2 to 0
                margin=dict(l=5, r=5, t=0, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)",
                ) if show_legend_once else None,
                # FIX: Removed the invalid 'titlefont' argument from the xaxis dict
                xaxis=dict(title="", showgrid=True, showticklabels=True, range=x_range, tickfont=dict(size=10)), # Added tick font size
                yaxis=dict(showticklabels=False),
            )

            st.plotly_chart(bar, config={'displayModeBar': False})

            show_legend_once = False # Hide legend for subsequent charts

        st.caption("This shows the average 'Current' vs 'Prescribed' value of each factor for the selected district(s).")


def plot_charts(change_df, df_original, df_modified, color_scale, map_title, bar_title):
    plot_df = change_df.copy()
    geo_key = st.session_state["geo_unit_col"]
    map_col = "Change"
    bar_col = "Change"

    GEO_COL = st.session_state["GEO_COL"]
    gdf = st.session_state["gdf"]

    # Merge shapefile with change data
    plot_df = gdf.merge(plot_df, left_on=GEO_COL, right_on=geo_key, how="left").copy()
    # Filter out null values which cannot be plotted
    plot_df = plot_df[plot_df["Change"].notnull()]
    if "prescriptive_detail_plot_df" not in st.session_state:
        st.session_state["prescriptive_detail_plot_df"]=plot_df
        st.session_state["prescriptive_detail_color_scale"]=color_scale
        st.session_state["prescriptive_detail_df_original"]=df_original
        st.session_state["prescriptive_detail_df_modified"]=df_modified
        # 1. Initialize the session state variable if it doesn't exist


    # 2. Check if a selection has been made on the map chart
    if "district_map_chart" in st.session_state:
        # Get the names of the newly selected districts from the chart
        newly_selected_districts = extract_district_names(st.session_state.get('district_map_chart'))

        # if(newly_selected_districts)
        st.session_state["currently_selected_districts"].extend(newly_selected_districts)

        map_fig = create_map_figure(plot_df, color_scale, st.session_state["currently_selected_districts"])

    else:
        # No map selection yet, create the default map
        map_fig = create_map_figure(plot_df, color_scale,st.session_state["currently_selected_districts"])


    # --- Bar Chart Creation ---
    bar_fig = create_bar_figure(plot_df, color_scale, bar_title, st.session_state["currently_selected_districts"])


    # --- PLOTLY CONFIG FOR SELECTION ---
    plotly_config = {
        'displayModeBar': True,
        'autoScale2d':True,
        "resetScale2d":True,
        'modeBarButtonsToRemove': [
            'zoom2d',
            'hoverClosestGeo', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines'
        ],

    }
    # ------------------------------------

    # Change to 3 columns: Map, Bar Chart, Feature Comparison Charts
    col1, col2, col3 = st.columns(3)

    # 1. Map Display and Selection List
    with col1:
        st.markdown(f"#### District Map")
        # Map container
        with st.container(height=380):
            # Display map and use callback for selection logic.
            # Relying on on_select callback to update session state for subsequent reruns.
            # if "district_map_chart" not in st.session_state or st.session_state["district_map_chart"]==None:
            st.plotly_chart(
                    map_fig,

                    key="district_map_chart",
                    on_select="rerun",
                    selection_mode=["box","lasso"],
                    config=plotly_config
                )
            st.session_state["prescriptive_displayed_main_map"]=map_fig


    # 2. Bar Chart
    with col2:
        st.markdown(f"#### Target Change by District")
        with st.container(height=380):
            st.plotly_chart(bar_fig)
            st.session_state["prescriptive_displayed_main_bar"]=bar_fig


# 3. Factor Comparison Charts
    with col3:
        st.markdown(f"#### Factor Comparison")

        create_feature_comparison_charts(df_original, df_modified, st.session_state["currently_selected_districts"])











    col1, col2, col3 = st.columns(3)
    with col1:
        if len(st.session_state["currently_selected_districts"])>0:
            # st.session_state["currently_selected_districts"]=extract_district_names(st.session_state.get('district_map_chart'))
            st.info(f"Currently Selected Districts:{st.session_state['currently_selected_districts']}")
        else:
            st.info("Use the **Box** or **Lasso** select tool on the map to choose districts. The feature comparison chart will update to show the average of these districts.")
    with col2:
        if st.button("Clear Districts Selections"):
            st.session_state['currently_selected_districts']=[]
            st.rerun()

    with col3:
        # Function to toggle the full-page view
        if 'show_big_popover' not in st.session_state:
            st.session_state['show_big_popover']=False
        def toggle_popover():
            st.session_state['show_big_popover'] = not st.session_state['show_big_popover']

        # Button to trigger the 'popover'
        if st.button("See District Wise Prescriptions", on_click=toggle_popover):
            pass # Action is handled by on_click




def run_prescriptive_modelling():
    st.set_page_config(
        initial_sidebar_state="collapsed"
    )
    if 'show_big_popover' not in st.session_state:
        st.session_state['show_big_popover']=False

    if st.session_state['show_big_popover']:
        run_detail_prescriptions_view(st.session_state["prescriptive_detail_plot_df"],
                                      st.session_state["prescriptive_detail_color_scale"],
                                      st.session_state["prescriptive_detail_df_original"],
                                      st.session_state["prescriptive_detail_df_modified"],
                                      st.session_state['currently_selected_districts'])
    else:
        st.title("📋 Prescriptive Modelling")

        # Assuming these session states are populated elsewhere
        trained_models = st.session_state.get("trained_models", {})
        feature_importances = st.session_state.get("feature_importances", {})

        # Check for required session states before proceeding
        if "geo_unit_col" not in st.session_state or \
                "target_col" not in st.session_state or \
                "grouped_df" not in st.session_state:
            st.error("⚠️ Required data or configuration missing. Please ensure data is loaded and models are trained.")
            st.stop()

        geo_unit_col = st.session_state["geo_unit_col"]
        target_col = st.session_state["target_col"]
        direction = st.session_state.get("target_direction", "Increase")
        positive_indicators = st.session_state.get("final_positive", [])
        negative_indicators = st.session_state.get("final_negative", [])
        gdf = st.session_state.get("gdf") # Made gdf optional, as it is only used for plotting
        df = st.session_state["grouped_df"].copy()

        if not trained_models:
            st.error("⚠️ Please train models in the Home tab first.")
            st.stop()

        # --- TARGET INITIALIZATION ---
        if "target_val" not in st.session_state:
            mean_val = np.round(df[target_col].mean(),2)
            if direction == 'Increase':
                st.session_state["target_val"] = float(mean_val + 5)
            else:
                st.session_state["target_val"] = float(mean_val - 5)

        # --- SIDE-BY-SIDE LAYOUT for Model Selection and Target Indicator ---
        col_model, col_target = st.columns(2)

        with col_model:
            # Added H3 markdown title to match the Target Indicator section
            st.markdown("### 🤖 Selected Model")
            model_names = list(trained_models.keys())
            # Model Selection Dropdown: label_visibility is now collapsed
            selected_model_name = st.selectbox(
                "🔍 Select a trained model",
                model_names,
                label_visibility="collapsed",
                key="model_select_input"
            )

        with col_target:
            st.markdown("### 🎯 Target Value")

            # Number input that persists for the target value
            # CORRECTION 1: Replaced "" with " " for accessibility
            target_val = st.number_input(
                " ",
                value=st.session_state["target_val"],  # Load from session state
                format="%.2f",
                label_visibility="collapsed",
                key="target_val_input"  # unique key
            )

        # Update session state whenever user changes value
        st.session_state["target_val"] = target_val

        # --- MODEL & FEATURE SETUP (Continue original flow) ---
        model = trained_models[selected_model_name]
        importances = feature_importances[selected_model_name]
        selected_feats = list(importances.index)

        # Initialize the filtered feature list in session state
        if "prescriptive_filtered_feats" not in st.session_state:
            st.session_state["prescriptive_filtered_feats"] = selected_feats.copy()

        avg_target = np.round(df[target_col].mean(),2)

        df_mod = df.copy()
        # Ensure scaler and feature_bounds are available
        scaler = st.session_state.get("scaler", None)
        if scaler is None and selected_model_name == "linear":
            st.error("⚠️ Linear model selected but scaler object is missing from session state.")
            st.stop()

        feature_order = st.session_state.get("trained_feature_names", selected_feats)

        if selected_model_name == "linear":
            # means and stds are not directly used in the prescriptive calculation logic, but kept for context
            # means = dict(zip(feature_order, scaler.mean_))
            # stds = dict(zip(feature_order, scaler.scale_))

            if "final_feature_importances" in st.session_state:
                # User-edited importances
                importances_df = st.session_state["final_feature_importances"][selected_model_name]
                importances_dict = importances_df.set_index("Feature")["Importance"].to_dict()
            else:
                # Use model's coefficients
                importances_dict = dict(zip(feature_order, model.coef_))
        else:
            # Placeholder for non-linear models
            importances_dict = {}

        if "sensitivities" not in st.session_state:
            st.session_state["sensitivities"] = {}

        # Ensure feature_bounds exists
        if "feature_bounds" not in st.session_state:
            st.session_state.feature_bounds = {}

        # ------------------ CORE PRESCRIPTIVE CALCULATION (KEPT AT TOP FOR LOGIC) ------------------

        # 1. Ensure sensitivities are set for all features
        for feat in selected_feats:
            if feat not in st.session_state["sensitivities"]:
                st.session_state["sensitivities"][feat] = 0.5 # Default sensitivity

        # 2. Apply adjustments to df_mod based on current sensitivities
        for feat in selected_feats:
            sens = st.session_state["sensitivities"][feat]
            fi = importances.get(feat, 0.001)

            if target_val == avg_target:
                delta = pd.Series(0, index=df.index)
            else:
                # Note: The original code used target_val - df[feat].
                # This 'delta' is the gap between the target and the current feature value.
                # This logic might be unconventional for prescriptive models but kept for consistency.
                delta = target_val - df[feat]

            adjustment = 0.0

            if selected_model_name == "linear":
                coef = importances_dict.get(feat, 0.001)
                if abs(coef) < 1e-6:
                    # Skipping near-zero coefs, df_mod[feat] remains df[feat] (the copy)
                    continue

                # Determine adjustment based on direction and indicator type
                if direction == "Increase":
                    # To Increase target, for POSITIVE coefs, we need to DECREASE the feature (hence the -1 * abs)
                    if feat in positive_indicators:
                        adjustment = -1 * abs(sens * (delta / coef))
                    # To Increase target, for NEGATIVE coefs, we need to INCREASE the feature
                    elif feat in negative_indicators:
                        adjustment = abs(sens * (delta / coef))
                elif direction == "Decrease":
                    # To Decrease target, for POSITIVE coefs, we need to DECREASE the feature
                    if feat in positive_indicators:
                        adjustment = abs(sens * (delta / coef))
                    # To Decrease target, for NEGATIVE coefs, we need to INCREASE the feature (hence the -1 * abs)
                    elif feat in negative_indicators:
                        adjustment = -1 * abs(sens * (delta / coef))

            else: # Non-linear models (using Feature Importance as a proxy for effect magnitude)
                # The logic here is highly simplified for non-linear models, assuming FI acts like a coefficient magnitude.
                if direction == "Increase":
                    if feat in positive_indicators:
                        adjustment = -1 * abs(sens * (delta / fi))
                    elif feat in negative_indicators:
                        adjustment = abs(sens * (delta / fi))
                elif direction == "Decrease":
                    if feat in positive_indicators:
                        adjustment = abs(sens * (delta / fi))
                    elif feat in negative_indicators:
                        adjustment = -1 * abs(sens * (delta / fi))

            # Apply the adjustment
            new_val_series = df[feat] - adjustment

            # Apply bounds
            min_bound, max_bound = st.session_state.feature_bounds.get(feat, (float('-inf'), float('inf')))
            df_mod[feat] = new_val_series.clip(lower=min_bound, upper=max_bound)

        # ------------------ END CORE PRESCRIPTIVE CALCULATION ------------------


        # ------------------- SCOREBOARDS, MAP AND BAR CHART -------------------

        if direction == "Increase":
            color_scale =  "RdYlGn"
        elif direction == "Decrease":
            color_scale = "RdYlGn_r"
        else:
            color_scale = "Viridis"

        # Predict with original and modified data
        if selected_model_name == "linear":
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

        # If Decrease and change > 0, reset to actual (can't increase when trying to decrease)
        if direction == 'Decrease':
            df_mod["Predicted"] = np.where(change > 0, df[target_col], df_mod["Predicted"])

        # If Increase and change < 0, reset to actual (can't decrease when trying to increase)
        elif direction == 'Increase':
            df_mod["Predicted"] = np.where(change < 0, df[target_col], df_mod["Predicted"])

        df_mod["Change"] = df_mod["Predicted"] - df[target_col]


        if 'currently_selected_districts' not in st.session_state:
            st.session_state['currently_selected_districts'] = []
        # --- Scoreboards ---
        # MODIFICATION: Changed columns from (3) to a narrower width for compactness
        col1, col2, col3 = st.columns(3)

        # MODIFICATION: Reduced padding, font sizes, and margins in inline HTML for compactness
        scoreboard_style = "background-color:#f9ecd8;padding:6px 8px;border-radius:6px;text-align:center;height:90px;"
        title_style = "font-size:15px;font-weight:600;margin-bottom:1px;line-height:1.2;"
        subtitle_style = "font-size:10px;color:#777;margin-bottom:3px;line-height:1.2;"
        value_style = "font-size:20px;font-weight:700;"

        with col1:
            st.markdown(f"""
            <div style="{scoreboard_style}">
                <div style="{title_style}">Current {target_col.replace("_", " ")}</div>
                <div style="{subtitle_style}">per District</div>
                <div style="{value_style}">{df[target_col].mean():.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="{scoreboard_style}">
                <div style="{title_style}">🎯 Target Value for {target_col.replace("_", " ")}</div>
                <div style="{subtitle_style}">set by user</div>
                <div style="{value_style}">{target_val:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            value_display_for_avg=df_mod['Change'].mean()
            if(len(st.session_state["currently_selected_districts"])>0):
                value_display_for_avg=df_mod[df_mod[geo_unit_col].isin(st.session_state["currently_selected_districts"])]['Change'].mean()

            st.markdown(f"""
            <div style="{scoreboard_style}">
                <div style="{title_style}">Average Change in {target_col.replace("_", " ")}</div>
                <div style="{subtitle_style}">per District</div>
                <div style="{value_style}">{value_display_for_avg:.2f}</div>
            </div>
            """, unsafe_allow_html=True)


        change_df = df_mod[[geo_unit_col, "Change"]].copy()
        plot_charts(
            change_df=change_df,
            df_original=df, # Pass original data
            df_modified=df_mod, # Pass modified data
            color_scale=color_scale,
            map_title="🗺️ District-wise Impact Visualization",
            bar_title="Impact by District"
        )




        # ------------------ MOVED UI: BUCKETS AND SENSITIVITIES ------------------

        st.markdown("---")
        st.markdown("### 🔧 Adjust Sensitivities & View Prescriptions")

        # Add bucket selection filter
        feature_buckets_grouped = st.session_state.get("feature_buckets_grouped", {})

        # Add an "Unassigned" bucket for features not in any existing bucket
        all_grouped_feats = set(sum(feature_buckets_grouped.values(), []))
        unassigned_feats = [f for f in selected_feats if f not in all_grouped_feats]
        feature_buckets_grouped["Unassigned"] = unassigned_feats

        bucket_names = list(feature_buckets_grouped.keys())
        bucket_names.sort()

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

        # # Display loop for sliders and comparison bars
        # show_legend = True

        # Use the filtered list of features for the main loop
        for feat in st.session_state.get("prescriptive_filtered_feats", selected_feats):
            col1, col2 = st.columns([1.4, 2.6])

            # Recalculate mean values for display based on df and df_mod
            old_val = df[feat].mean()
            # Use the value calculated in the core logic block after clipping to bounds
            new_val = df_mod[feat].mean()


            show_legend = False

        # st.caption("⬅️ Adjust sliders to shift factor values. Right bars show new vs current average across districts.")

        # Show in Streamlit

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
        st.dataframe(display_df[display_df[geo_unit_col].isin(st.session_state["currently_selected_districts"])])

        # ------------------ DOWNLOAD EXPANDER ------------------

        with st.expander("📤 Download Prescriptions"):
            format_choice = st.radio("Choose format:", ["CSV", "Excel"], horizontal=True)

            if format_choice == "CSV":
                csv = display_df.to_csv(index=False).encode('utf-8')
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
