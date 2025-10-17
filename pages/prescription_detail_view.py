import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as colors
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import io
import json
from functools import partial

# --- Utility Functions ---
import streamlit as st
import plotly.graph_objects as go

# --- 1. Custom Legend Function ---

import streamlit as st

def create_fixed_legend():
    """
    Creates a fixed horizontal legend for 'Current' (steelblue) and
    'Prescribed' (deepskyblue) with clear color boxes.
    """
    # Define the fixed items
    legend_items = [
        {'name': 'Current', 'color': 'steelblue'},
        {'name': 'Prescribed', 'color': 'deepskyblue'}
    ]

    # Use st.container to ensure a clean grouping
    with st.container():
        # Use st.columns to lay out the legend items horizontally
        # Adjusted ratio to give enough space for the items
        cols = st.columns([2, 2, 6])

        for i, item in enumerate(legend_items):
            color = item['color']
            name = item['name']

            # Use a robust, single-line HTML structure for the box and text
            # Setting 'display: inline-flex' and 'vertical-align: middle' helps
            # make sure the box and text align correctly.
            legend_html = f"""
            <div style="display: inline-flex; align-items: center; white-space: nowrap; margin-right: 15px;">
                <span style="
                    display: inline-block;
                    width: 12px; 
                    height: 12px; 
                    background-color: {color}; 
                    border: 1px solid #777; /* Added a slight border for visibility */
                    margin-right: 6px;
                    border-radius: 2px;
                "></span>
                <span style="font-size: 14px; color: #333; line-height: 1.2;">{name}</span>
            </div>
            """

            # Render the styled HTML in the respective column
            with cols[i]:
                st.markdown(legend_html, unsafe_allow_html=True)

# Example usage (if you run this file directly)
# create_fixed_legend()

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

def update_district_selections_detail():
    # 2. Check if a selection has been made on the map chart

    newly_selected_districts = extract_district_names(st.session_state.get('district_map_chart_detail'))

    # If new districts were selected via the map, append them.
    # Note: The logic in run_detail_prescriptions_view for `on_select` is more complex,
    # but for simplicity and to match the original intent, we'll keep this function minimal.
    # The selection logic in run_detail_prescriptions_view might overwrite this.
    # For now, we'll ensure we only add new ones to avoid duplicates if the list is being managed elsewhere.
    current_list = st.session_state.get("currently_selected_districts", [])
    for district in newly_selected_districts:
        if district not in current_list:
            current_list.append(district)

    st.session_state["currently_selected_districts"] = current_list
    # st.rerun()

# --- Map Creation Functions (Corrected) ---

def create_mini_map_figure(plot_df, color_scale, selected_district):

    """
    Creates and returns the Plotly choropleth map figure,
    zoomed to the area of the data in the GeoDataFrame.
    """

    # Retrieve necessary variables from session state
    # Added placeholders for robustness in a non-full app context
    if "geo_unit_col" not in st.session_state or "gdf" not in st.session_state:
        st.session_state["geo_unit_col"] = 'district' # Placeholder
        st.session_state["GEO_COL"] = 'geometry' # Placeholder

    geo_key = st.session_state["geo_unit_col"]
    GEO_COL = st.session_state["GEO_COL"] # Should be 'geometry'

    map_col = "Change" # Using "Change" based on your usage

    # If the plot_df is empty or multi-row, we take the first district for reference.
    plot_df_filtered = plot_df[plot_df[geo_key] == selected_district]

    if plot_df_filtered.empty:
        return go.Figure().update_layout(title=f"No data for {selected_district}", height=250)

    # Use the filtered data for color and bounds
    color_min = plot_df["Change"].min()
    color_max = plot_df["Change"].max()

    # --- Bounding Box Calculation (Focus on the single district's bounds) ---



    # FIX 1: Use .iloc[0] to access the scalar value from the Series
    optimal_zoom = float(plot_df_filtered["_optimal_zoom"].iloc[0]) if "_optimal_zoom" in plot_df_filtered.columns else 8


    # --- Choropleth Map (Base Layer) ---
    map_fig = px.choropleth_mapbox(
        plot_df_filtered,
        geojson=plot_df_filtered.__geo_interface__,
        locations=plot_df_filtered.index,
        color=map_col,
        hover_name=geo_key,
        hover_data={
            map_col: ':.2f',
        },
        mapbox_style="white-bg",
        opacity=0.7,
        color_continuous_scale=color_scale,
        range_color=[color_min, color_max],
        labels={map_col: 'Change'},
        center={"lat": float(plot_df_filtered.geometry.unary_union.centroid.y), "lon":float(plot_df_filtered.geometry.unary_union.centroid.x)},
        zoom=optimal_zoom
    )

    # --- Highlighting Trace (Labels) ---
    map_fig.add_trace(go.Scattermapbox(
        lon=[g.centroid.x for g in plot_df_filtered.geometry],
        lat=[g.centroid.y for g in plot_df_filtered.geometry],
        mode='text',
        # This .apply logic is fine, but ensures float conversion is explicit
        text=plot_df_filtered.apply(lambda row: f"{row[geo_key]}<br>{float(row['Change']):.2f}", axis=1),
        textfont=dict(size=10, color='black', family="Inter"),
        hoverinfo='none',
        showlegend=False
    ))

    # Remove Index from Hover Tooltip using hovertemplate
    if map_fig.data:
        map_fig.data[0].hovertemplate = f'<b>%{{hovertext}}</b><br>Change: %{{z:.2f}}<extra></extra>'

    # FIX 2: Ensure height is set in the figure layout, not the st.plotly_chart call
    map_fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_showscale=False,
        height=250 # <-- Set height here
    )

    return map_fig

def create_map_figure(plot_df, color_scale, highlight_districts_list=None):
    """
    Creates and returns the Plotly choropleth map figure,
    with an option to highlight specific districts.
    """
    if highlight_districts_list is None:
        highlight_districts_list = []

    # Added placeholders for robustness in a non-full app context
    if "geo_unit_col" not in st.session_state or "gdf" not in st.session_state:
        st.session_state["geo_unit_col"] = 'district' # Placeholder
        st.session_state["GEO_COL"] = 'geometry' # Placeholder

    geo_key = st.session_state["geo_unit_col"]
    map_col = "Change"
    GEO_COL = st.session_state["GEO_COL"]
    # Assuming gdf is available and has the 'geometry' column for total bounds
    gdf = st.session_state.get("gdf", plot_df)



    # --- Choropleth Map (Base Layer) ---
    map_fig = px.choropleth_mapbox(
        plot_df,
        geojson=plot_df.__geo_interface__,
        locations=plot_df.index,
        color=map_col,
        hover_name=geo_key,
        hover_data={
            map_col: ':.2f',
        },
        mapbox_style="white-bg",
        opacity=0.7,
        color_continuous_scale=color_scale,
        labels={map_col: 'Change'},
        center={"lat": float(plot_df.geometry.unary_union.centroid.y), "lon":float(plot_df.geometry.unary_union.centroid.x)},
        zoom=4
    )

    # --- Highlighting Trace ---
    if highlight_districts_list and not plot_df.empty:
        highlight_df = plot_df[plot_df[geo_key].isin(highlight_districts_list)]

        if not highlight_df.empty:
            map_fig.add_trace(go.Choroplethmapbox(
                geojson=highlight_df.__geo_interface__,
                locations=highlight_df.index,
                z=[0] * len(highlight_df),
                colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
                marker_line_width=3,
                marker_line_color='red',
                hoverinfo='text',
                text=highlight_df[geo_key],
                name="Highlighted Districts",
                showlegend=False,
                showscale=False
            ))

    # Conditional Label Trace
    if 'geometry' in plot_df.columns and len(plot_df) > 0:
        map_fig.add_trace(go.Scattermapbox(
            lon=[g.centroid.x for g in plot_df.geometry],
            lat=[g.centroid.y for g in plot_df.geometry],
            mode='text',
            text=plot_df.apply(lambda row: f"{row[geo_key]}<br>{row[map_col]:.2f}", axis=1),
            textfont=dict(size=8, color='black'),
            hoverinfo='none',
            showlegend=False
        ))

    if map_fig.data:
        map_fig.data[0].hovertemplate = f'<b>%{{hovertext}}</b><br>Change: %{{z:.2f}}<extra></extra>'

    # map_fig.update_mapboxes(
    #     bounds={
    #         "west": float(minx)-10,
    #         "east": float(maxx)+10,
    #         "south": float(miny),
    #         "north": float(maxy)
    #     }
    # )

    # FIX 2: Ensure height is set in the figure layout
    map_fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=280 # Set height for the main map
    )

    return map_fig

def create_feature_districts_comparison_charts(df, df_mod, selected_districts):
    """
    Creates a set of dual bar charts comparing current vs. prescribed values for selected features
    across the currently selected districts, or the overall average if none are selected.
    """
    # Added placeholder for robustness in a non-full app context
    if "geo_unit_col" not in st.session_state:
        st.session_state["geo_unit_col"] = 'district'

    filtered_feats = st.session_state.get("prescriptive_filtered_feats", [])
    geo_key = st.session_state["geo_unit_col"]

    if not filtered_feats:
        st.caption("No features selected for prescriptive modelling.")
        return

    # --- MODIFIED LOGIC: Use all data if no districts are selected ---
    if selected_districts:
        df_filtered = df[df[geo_key].isin(selected_districts)]
        df_mod_filtered = df_mod[df_mod[geo_key].isin(selected_districts)]
        chart_title_suffix = f" (Avg over {len(selected_districts)} districts)"
    else:
        df_filtered = df
        df_mod_filtered = df_mod
        chart_title_suffix = " (Overall Average)"

    with st.container(height=250):
        show_legend_once = False

        for feat in filtered_feats:
            if df_filtered.empty or df_mod_filtered.empty:
                old_val = 0
                new_val = 0
            else:
                old_val = df_filtered[feat].mean()
                new_val = df_mod_filtered[feat].mean()

            st.markdown(f"<div style='margin-top: 0px;font-size: 12px;'>**{feat.replace('_', ' ')}** {chart_title_suffix}</div>", unsafe_allow_html=True)

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

            max_val = max(old_val, new_val)
            x_range = [0, max_val * 1.2] if max_val > 0 else [0, 1]

            bar.update_layout(
                title=" ",
                barmode='group',
                height=80,
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
                xaxis=dict(title="", showgrid=True, showticklabels=True, range=x_range, tickfont=dict(size=10)),
                yaxis=dict(showticklabels=False),
            )

            st.plotly_chart(bar,key=f"detail_feature_duel_bar_{feat}_{selected_districts}", use_container_width=True, config={'displayModeBar': False})


            # FIX 3: Removed deprecated width/height arguments (relying on layout height and use_container_width)
            # st.plotly_chart(bar,key=f"detail_feature_duel_bar_{feat}_{selected_districts}", use_container_width=True, config={'displayModeBar': False})

            show_legend_once = False

        st.caption("This shows the average 'Current' vs 'Prescribed' value of each factor for the selected district(s).")
def create_feature_all_comparison_charts(df, df_mod, selected_districts):
    """
    Creates a set of dual bar charts comparing current vs. prescribed values for selected features
    across the currently selected districts, or the overall average if none are selected.
    """
    # Added placeholder for robustness in a non-full app context
    if "geo_unit_col" not in st.session_state:
        st.session_state["geo_unit_col"] = 'district'

    filtered_feats = st.session_state.get("prescriptive_filtered_feats", [])
    geo_key = st.session_state["geo_unit_col"]

    if not filtered_feats:
        st.caption("No features selected for prescriptive modelling.")
        return

    # --- MODIFIED LOGIC: Use all data if no districts are selected ---
    if selected_districts:
        df_filtered = df[df[geo_key].isin(selected_districts)]
        df_mod_filtered = df_mod[df_mod[geo_key].isin(selected_districts)]
        chart_title_suffix = f" (Avg over {len(selected_districts)} districts)"
    else:
        df_filtered = df
        df_mod_filtered = df_mod
        chart_title_suffix = " (Overall Average)"

    with st.container(height=250):
        show_legend_once = False

        for feat in filtered_feats:
            if df_filtered.empty or df_mod_filtered.empty:
                old_val = 0
                new_val = 0
            else:
                old_val = df_filtered[feat].mean()
                new_val = df_mod_filtered[feat].mean()

            st.markdown(f"<div style='margin-top: 0px;font-size: 12px;'>**{feat.replace('_', ' ')}** {chart_title_suffix}</div>", unsafe_allow_html=True)

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

            max_val = max(old_val, new_val)
            x_range = [0, max_val * 1.2] if max_val > 0 else [0, 1]

            bar.update_layout(
                title=" ",
                barmode='group',
                height=80,
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
                xaxis=dict(title="", showgrid=True, showticklabels=True, range=x_range, tickfont=dict(size=10)),
                yaxis=dict(showticklabels=False),
            )

            st.plotly_chart(bar,key=f"detail_feature_duel_bar_{feat}_all_districts", use_container_width=True, config={'displayModeBar': False})


            # FIX 3: Removed deprecated width/height arguments (relying on layout height and use_container_width)
            # st.plotly_chart(bar,key=f"detail_feature_duel_bar_{feat}_{selected_districts}", use_container_width=True, config={'displayModeBar': False})

            show_legend_once = False

        st.caption("This shows the average 'Current' vs 'Prescribed' value of each factor for the selected district(s).")


def run_detail_prescriptions_view(plot_df, color_scale, df_original, df_modified, selected_districts):
    """
    Main layout function for the detail view, incorporating the map and comparison charts.
    Handles map selection updates via Streamlit's `on_select` callback mechanism.
    """
    map_fig = create_map_figure(plot_df, color_scale, selected_districts)

    col_left, col_right = st.columns([0.4,0.6])

    # --- Plotly Config ---
    plotly_config = {
        'displayModeBar': True,
        'modeBarButtonsToRemove': [
            'zoom2d', 'autoScale2d', 'resetScale2d',
            'hoverClosestGeo', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines'
        ],
    }

    # 2. Populate the Left Column (Main Map and Charts)
    with col_left:
        st.markdown(f"#### District Map (Click and drag to select)")

        with st.container(height=280):
            # Display map and use callback for selection logic.
            # FIX 3: Removed deprecated height argument. Height is set in map_fig layout.
            map_selection = st.plotly_chart(
                map_fig,
                # height=280, # REMOVED (now in map_fig.update_layout)
                key="district_map_chart_detail",
                selection_mode=["box","lasso"],
                on_select=update_district_selections_detail,
                config=plotly_config
            )
        if st.button("Clear Districts Selections"):
            st.session_state['currently_selected_districts']=[]
            st.rerun()

        # Handle Plotly selection: Update the list of selected districts
        # This logic is intended to run AFTER update_district_selections_detail (on_select)
        # However, st.plotly_chart with on_select might not fully populate
        # st.session_state["plotly_selection"] directly.
        # We'll trust the on_select callback to manage the list for now.
        selected_district_names = st.session_state.get('currently_selected_districts', [])

        st.markdown(f"#### Target Feature Comparison")
        create_feature_all_comparison_charts(df_original, df_modified, selected_district_names)


    # --- 3. Populate the Right Column (Mini Maps) ---

    current_selected_districts = st.session_state.get('currently_selected_districts', [])
    if(len(current_selected_districts)==0):
        # Fallback to show a sample if nothing is selected (assuming plot_df is a GeoDataFrame)
        # Note: Added check for GEO_COL presence
        geo_col_name = st.session_state.get("geo_unit_col", "district")
        if not plot_df.empty and geo_col_name in plot_df.columns:
            current_selected_districts=plot_df[geo_col_name].head(5).tolist()
        else:
            # Handle case where plot_df is empty or session state is not fully set up
            current_selected_districts = []


    with col_right:
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader(f"Selected Districts ({len(current_selected_districts)})")
        with col_right:
            def toggle_popover():
                # Assuming this popover logic is handled elsewhere
                st.session_state['show_big_popover'] = not st.session_state.get('show_big_popover', False)

            if st.button("GO back to Main dashbaord", on_click=toggle_popover):
                pass

        with st.container(height=700):
            if len(current_selected_districts) > 0:
                geo_col_name = st.session_state.get("geo_unit_col", "district")
                for district in current_selected_districts:
                    mini_map_df = plot_df[plot_df[geo_col_name] == district]

                    with st.container(border=True):
                        col_left, col_right = st.columns(2)
                        with col_left:
                            st.markdown(f"**{district}** 📍")
                        with col_right:
                            create_fixed_legend()

                        col_left, col_right = st.columns(2)

                        with col_left:
                            # Display the mini-map
                            # FIX 3: Removed deprecated width/height arguments. Height is set in the figure layout.
                            st.plotly_chart(
                                create_mini_map_figure(plot_df, color_scale, district),
                                use_container_width=True,
                                # width='stretch', # REMOVED
                                # height=250, # REMOVED (now in figure layout)
                                key=f"district_mini_map_chart_detail_{district}",
                                config=plotly_config
                            )
                        with col_right:
                            create_feature_districts_comparison_charts(df_original, df_modified, [district])

            else:
                st.info("Use the main map on the left to select districts by drawing a box or lasso around them.")