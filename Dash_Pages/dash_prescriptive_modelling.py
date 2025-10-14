import socket
import json
import pandas as pd
from dash import Dash, html, dash_table, dcc, callback_context, no_update
from dash.dependencies import Input, Output, State, ALL
from werkzeug.serving import run_simple
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
from typing import Dict, Any, List, Tuple
from dash.dash_table import Format
import io

from Dash_Pages.dash_prescription_detail import generate_detail_page_layout, register_detail_page_callbacks

# --- IMPORT MODULARIZED DETAIL PAGE COMPONENTS ---
# ------------------------------------------------

# ==============================================================================
# 1. SHARED GLOBAL STATE AND UTILITIES
# ==============================================================================

# GLOBAL LOOKUP TABLE: This dictionary serves as the shared memory space
# where non-serializable Python objects (like trained models, scalers,
# and GeoDataFrames) are stored by Streamlit (main thread) and retrieved
# by Dash (server thread).
PYTHON_OBJECT_LOOKUP = {}



# ==============================================================================
# 2. CORE CALCULATION AND PLOT GENERATION LOGIC (Shared Utilities)
# ==============================================================================

def calculate_impact(
        df: pd.DataFrame,
        app_state: Dict[str, Any],
        target_val: float,
        sensitivities: Dict[str, float],
        selected_model_name: str
) -> pd.DataFrame:
    """
    Performs the prescriptive impact analysis using the actual trained model, aiming to
    shift predictions towards the user-defined target_val by adjusting features
    based on their sensitivity and coefficients/importances.
    """
    df_mod = df.copy()
    target_col = app_state.get("target_col", "N/A")
    sig_feats = app_state.get("sig_feats", [])
    direction = app_state.get("target_direction", "Increase")
    positive_indicators = app_state.get("final_positive", [])
    negative_indicators = app_state.get("final_negative", [])
    feature_bounds = app_state.get("feature_bounds", {})
    target_bounds = app_state.get("target_bounds", (float("-inf"), float("inf")))
    feature_importances = app_state.get("final_feature_importances", {})


    # --- 1. Retrieve Model, Scaler, and Feature Order from shared state ---
    trained_models_refs = app_state.get("trained_models", {})
    scaler_id = app_state.get("scaler")
    feature_order = app_state.get("trained_feature_names", sig_feats) # The features used for training

    scaler = PYTHON_OBJECT_LOOKUP.get(scaler_id)
    model_id = trained_models_refs.get(selected_model_name)
    model = PYTHON_OBJECT_LOOKUP.get(model_id)

    if model is None or not feature_order or target_col not in df.columns:
        # Fallback if the necessary components are missing
        df["Predicted"] = df[target_col] if target_col in df.columns else 0.0
        df_mod["Predicted"] = df["Predicted"]
        df_mod["Change"] = 0.0
        if target_col in df.columns:
            df_mod[target_col] = df[target_col]
        return df_mod

    # --- 2. Setup Prescriptive Parameters ---
    # Get feature importances/coefficients based on model type
    if selected_model_name == "linear":
        if "final_feature_importances" in app_state:
            importances_df = PYTHON_OBJECT_LOOKUP["final_feature_importances"][selected_model_name]
            importances_dict = importances_df.set_index("Feature")["Importance"].to_dict()
        else:
            # Use model's coefficients
            if hasattr(model, 'coef_'):
                importances_dict = dict(zip(feature_order, model.coef_))
            else:
                importances_dict = {} # Should not happen for linear model
        importances = importances_dict
    else:
        # Use provided feature importances for non-linear models
        importances = feature_importances.get(selected_model_name, {})

    # Start with a copy of original feature values for prediction input
    df_to_predict = df[feature_order]
    df_mod_to_predict = df[feature_order].copy()


    # --- 3. CORE PRESCRIPTIVE FEATURE ADJUSTMENT ---
    for feat in feature_order:
        sens = sensitivities.get(feat, 0.5) # Use user's input sensitivity (default 0.5)
        fi = importances.get(feat, 0.001)

        # Calculate the delta required from the current instance value
        delta = target_val - df[target_col]

        adjustment = pd.Series(0, index=df.index)

        if abs(fi) < 1e-6:
            # Skip features with near-zero importance/coefficient
            continue

        if selected_model_name == "linear":
            # Linear model logic uses coefficient (fi = coef)
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

        else: # Non-linear models (using feature importance as surrogate)
            # Replicating provided logic for non-linear models:
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


        # Apply the calculated adjustment to the feature value
        new_val_series = df[feat] - adjustment

        # Apply bounds to the adjusted feature values
        min_bound, max_bound = feature_bounds.get(feat, (float('-inf'), float('inf')))
        df_mod_to_predict[feat] = new_val_series.clip(lower=min_bound, upper=max_bound)


    # --- 4. Perform Prediction (Baseline and Modified) ---

    # Predict with original data (Baseline)
    if selected_model_name == "linear" and hasattr(scaler, 'transform'):
        df_scaled = scaler.transform(df_to_predict)
        df["Predicted"] = model.predict(df_scaled)
    else:
        df["Predicted"] = model.predict(df_to_predict)

    # Predict with modified data
    if selected_model_name == "linear" and hasattr(scaler, 'transform'):
        df_mod_scaled = scaler.transform(df_mod_to_predict)
        df_mod["Predicted"] = model.predict(df_mod_scaled)
    else:
        df_mod["Predicted"] = model.predict(df_mod_to_predict)


    # --- 5. Final Adjustment and Change Calculation ---

    # Calculate delta (difference between modified prediction and baseline prediction)
    delta_pred = df_mod["Predicted"] - df["Predicted"]

    # Calculate the modified target value: Original Target + Delta
    df_mod["Predicted"] = df[target_col] + delta_pred

    # Apply bounds to the final prediction
    t_min, t_max = target_bounds
    df_mod["Predicted"] = df_mod["Predicted"].clip(lower=t_min, upper=t_max)

    # Calculate the Change based on the bounded prediction
    df_mod["Change"] = df_mod["Predicted"] - df[target_col]

    # Re-merge the modified feature columns back into df_mod
    for feat in feature_order:
        df_mod[feat] = df_mod_to_predict[feat]

    # Store the original target for display purposes
    df_mod[target_col] = df[target_col]
    return df_mod

def generate_bar_chart(change_df: pd.DataFrame, app_state: dict) -> px.bar:
    """
    Generates the District Impact Summary Bar Chart (showing 'Change').
    """
    plot_df = change_df.copy()
    geo_key = app_state.get("geo_unit_col", "N/A")
    bar_col = "Change"

    direction = app_state.get("target_direction", "Increase")

    if direction == "Increase":
        color_scale = "RdYlGn"
    elif direction == "Decrease":
        color_scale = "RdYlGn_r"
    else:
        color_scale = "Viridis"

    # --- Start Robust geo_key handling ---
    if geo_key not in plot_df.columns:
        if 'District' in plot_df.columns:
            geo_key = 'District'
        else:
            geo_key = "N/A"
    # --- End Robust geo_key handling ---

    # --- Data Cleaning ---
    if geo_key == "N/A" or plot_df.empty:
        empty_bar = px.bar(title=None, height=450)
        return empty_bar

    # Filter out null values
    bar_df = plot_df[plot_df["Change"].notnull()].copy()

    if bar_df.empty:
        empty_bar = px.bar(title=None, height=450)
        return empty_bar

    # --- Bar Chart Logic ---
    # Sort by absolute change for better readability
    bar_df['AbsChange'] = bar_df[bar_col].abs()
    bar_df = bar_df.sort_values('AbsChange', ascending=False).copy()

    # Calculate dynamic height to ensure a fixed bar thickness of approximately 20px per bar.
    N_districts = len(bar_df[geo_key].unique())
    # Reduced height for detail page summary bar, as it's stacked
    dynamic_height = max(200, N_districts * 20 + 80)

    bar_fig = px.bar(
        bar_df,
        x=bar_col, y=geo_key, orientation='h',
        title=None, # Remove default title
        labels={geo_key: 'District', bar_col: 'Change'},
        color=bar_col,
        color_continuous_scale=color_scale,
        height=dynamic_height, # Use dynamic height for fixed bar thickness
        text=bar_df[bar_col].round(2),
        hover_data={
            geo_key: False,
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
        margin=dict(l=150, r=0, t=20, b=0),
        bargap=0.1,
        yaxis=dict(
            tickfont=dict(size=10),
        ),
        uirevision=True # Keep bar chart state consistent
    )

    return bar_fig

def generate_target_comparison_chart(impact_df: pd.DataFrame, selected_districts: List[str]) -> go.Figure:
    """
    Generates a duel bar chart comparing Original Target Value vs. Modified Predicted Value
    for the selected districts (the 'Prescriptions Duel Bar Chart').
    """
    if impact_df.empty or not selected_districts:
        return go.Figure(layout=dict(title="No Districts Selected for Detailed Comparison", height=450))

    # Filter data for selected districts
    chart_df = impact_df[impact_df['GeoUnitName'].isin(selected_districts)].copy()

    if chart_df.empty:
        return go.Figure(layout=dict(title="No Data for Selected Districts", height=450))

    # Melt the dataframe for grouped bar chart plotting
    plot_df = chart_df.melt(
        id_vars=['GeoUnitName'],
        value_vars=['Target (Original)', 'Predicted'],
        var_name='Metric',
        value_name='Value'
    )

    # Sort the districts by the predicted value for better visualization
    district_order = chart_df.sort_values('Predicted', ascending=False)['GeoUnitName'].tolist()

    # Dynamic height based on number of selected districts
    N_districts = len(district_order)
    dynamic_height = max(350, N_districts * 50 + 80) # Use 50px per district for vertical bars

    duel_fig = px.bar(
        plot_df,
        x='GeoUnitName',
        y='Value',
        color='Metric',
        barmode='group',
        text_auto='0.2f',
        title=None,
        labels={'GeoUnitName': 'District', 'Value': 'Target Value'},
        color_discrete_map={
            'Target (Original)': '#1d4ed8',  # Blue (Original)
            'Predicted': '#059669'         # Green (Modified/Predicted)
        },
        height=dynamic_height
    )

    # Apply district order
    duel_fig.update_xaxes(categoryorder='array', categoryarray=district_order)

    duel_fig.update_traces(
        textposition='outside',
        textfont=dict(size=10)
    )

    duel_fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=100), # Adjust bottom margin for rotated labels
        legend_title_text='Value Type',
        xaxis_tickangle=-45,
        plot_bgcolor='white',
        paper_bgcolor='white',
        uniformtext_minsize=8, uniformtext_mode='hide'
    )

    return duel_fig

def generate_map(change_df: pd.DataFrame, app_state: dict, gdf: gpd.GeoDataFrame | None, selected_districts: list | None = None) -> px.choropleth_mapbox:
    """
    Generates the Plotly Mapbox Choropleth map.
    """
    plot_df = change_df.copy()
    geo_key = app_state.get("geo_unit_col", "N/A")
    map_col = "Change"
    direction = app_state.get("target_direction", "Increase")
    GEO_COL = app_state.get("GEO_COL", "N/A")

    selected_districts = selected_districts or [] # Ensure it's a list

    if direction == "Increase":
        color_scale = "RdYlGn"
    elif direction == "Decrease":
        color_scale = "RdYlGn_r"
    else:
        color_scale = "Viridis"

    # --- Start Robust geo_key handling ---
    local_geo_key = geo_key
    if local_geo_key not in plot_df.columns:
        if 'district' in plot_df.columns:
            local_geo_key = 'district'
        elif 'District' in plot_df.columns:
            local_geo_key = 'District'
        else:
            return go.Figure(layout=dict(title=None, height=500, margin=dict(l=0, r=0, t=0, b=0)))
    # --- End Robust geo_key handling ---

    fixed_height = 500
    if gdf is None or gdf.empty or local_geo_key == "N/A":

        return go.Figure(layout=dict(
            title=None,
            height=fixed_height,
            margin=dict(l=0, r=0, t=0, b=0)
        ))

    # --- Merge shapefile with change data ---
    plot_df = gdf.merge(change_df[[local_geo_key, map_col]], left_on=GEO_COL, right_on=local_geo_key, how="left").copy()
    plot_df = plot_df[plot_df[map_col].notnull()]

    if plot_df.empty:
        return go.Figure(layout=dict(
            title=None,
            height=fixed_height,
            margin=dict(l=0, r=0, t=0, b=0)
        ))

    # --- Geographic Bounds and Centering ---
    # minx, miny, maxx, maxy = gdf.total_bounds
    center_lon = plot_df.geometry.centroid.x
    center_lat = plot_df.geometry.centroid.y




    # --- Calculate Centroids for Labels ---
    try:
        plot_df['centroid_lon'] = plot_df.geometry.centroid.x
        plot_df['centroid_lat'] = plot_df.geometry.centroid.y
    except Exception as e:
        # Fallback to mean lat/lon
        plot_df['centroid_lon'] = center_lon
        plot_df['centroid_lat'] = center_lat


    # --- Choropleth Map ---
    map_fig = px.choropleth_mapbox(
        plot_df,
        geojson=plot_df.__geo_interface__,
        locations=plot_df.index, # Use DataFrame index for selection tracking
        color=map_col,
        hover_name=local_geo_key,
        hover_data={
            map_col: ':.2f',
            local_geo_key: False,
        },
        mapbox_style="white-bg",
        opacity=0.7,
        color_continuous_scale=color_scale,
        labels={map_col: 'Change'},
        center={"lat": float(center_lat.mean()), "lon": float(center_lon.mean())},
        zoom=4
    )

    # --- Add Scattermapbox trace for labels ---
    map_fig.add_trace(go.Scattermapbox(
        lat=plot_df['centroid_lat'],
        lon=plot_df['centroid_lon'], # FIXED: Changed 'centroid_lat' to 'centroid_lon'
        mode='text',
        text=plot_df[local_geo_key] + " (" + plot_df[map_col].round(2).astype(str) + ")",
        textfont=dict(
            size=10,
            color="black"
        ),
        hoverinfo='skip',
        name='Labels',
        showlegend=False
    ))
    # --- END FIX ---

    # --- SELECTION PRESERVATION LOGIC ---
    if selected_districts and plot_df is not None and not plot_df.empty and map_fig.data:
        geo_key_for_select = local_geo_key
        # Use hover_name (district name) to find the index to select
        selected_indices = plot_df[plot_df[geo_key_for_select].isin(selected_districts)].index.tolist()

        if selected_indices:
            map_fig.data[0].selectedpoints = selected_indices
            map_fig.update_layout(selectionrevision=True)
    # --- END SELECTION PRESERVATION LOGIC ---

    map_fig.update_mapboxes(
        bearing=0,
        pitch=0,
        center={"lat": float(center_lat.mean())-3, "lon": float(center_lon.mean())},
        accesstoken=None, # Clear access token
        layers=[],
        zoom=4
    )

    map_fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title=None,
        height=fixed_height
    )

    return map_fig

def _prepare_data_table_columns(display_df: pd.DataFrame, geo_unit_col: str, intervene_feats: List[str], target_col_for_summary: str, new_predicted_col_name: str, new_change_col_name: str) -> List[Dict[str, Any]]:
    """
    Helper function to dynamically generate the column structure for the Dash DataTable.
    """
    columns = []
    # Added new_predicted_col_name to the list of columns that need numeric formatting
    summary_column_names_list = [new_change_col_name, new_predicted_col_name, target_col_for_summary]

    for i in display_df.columns:
        col_def = {"name": i, "id": i}
        # Apply numeric formatting only to numeric columns or explicit summary columns
        if display_df[i].dtype in [np.float64, np.float32, np.float16] or i in summary_column_names_list:
            col_def['type'] = 'numeric'
            col_def['format'] = Format.Format(precision=4, scheme=Format.Scheme.fixed)

        # Highlight intervention columns visually
        if any(f in i for f in intervene_feats) and ("(Original)" in i or "(Modified)" in i):
            col_def['style_header'] = {'backgroundColor': '#fef3c7', 'fontWeight': 'bold'}
            col_def['style_cell'] = {'backgroundColor': '#fffbeb'}

        columns.append(col_def)

    return columns


# ==============================================================================
# 3. CORE DASHBOARD LAYOUT (The main page layout remains here)
# ==============================================================================

def generate_main_dashboard_layout(
        initial_current_mean: float,
        initial_target_value: float,
        initial_bar_figure: go.Figure,
        initial_map_figure: go.Figure,
        initial_columns: List[Dict[str, Any]],
        initial_data: List[Dict[str, Any]],
        target_col: str,
        model_names: List[str],
        initial_selected_model: str,
        bucket_names: List[str],
        buckets_data_json: str,
        app_state: Dict[str, Any],
        initial_sens_map: Dict[str, float],
        initial_store_data_json: str,
        initial_display_data_json: str,
        initial_feature_averages_json: str,
        sig_feats: List[str]
) -> html.Div:
    """Generates the full layout for the main dashboard page (route /)."""
    return html.Div(
        [
            html.H1("Prescriptive Modelling Dashboard", style={'color': '#1e3a8a', 'border-bottom': '2px solid #60a5fa', 'padding-bottom': '10px'}),

            # --- Input Controls Row (Model Selection and Target Value Input) ---
            html.Div(
                style={'display': 'grid', 'grid-template-columns': 'repeat(auto-fit, minmax(300px, 1fr))', 'gap': '20px', 'margin-bottom': '20px'},
                children=[
                    # Model Selector
                    html.Div(
                        style={'background-color': '#ffffff', 'padding': '15px', 'border-radius': '8px', 'border': '1px solid #e5e7eb'},
                        children=[
                            html.H3("🔍 Model Selection", style={'color': '#1d4ed8', 'font-size': '1.5rem', 'margin-top': '0'}),
                            dcc.Dropdown(
                                id='model-selector-dropdown',
                                options=[{'label': name, 'value': name} for name in model_names],
                                value=initial_selected_model,
                                clearable=False,
                                style={'margin-bottom': '10px'}
                            ),
                            html.P(id='selected-model-output', style={'margin-top': '10px', 'font-weight': '500'})
                        ]
                    ),

                    # Target Value Input
                    html.Div(
                        style={'background-color': '#ffffff', 'padding': '15px', 'border-radius': '8px', 'border': '1px solid #e5e7eb'},
                        children=[
                            html.H3("🎯 Target Value Input", style={'color': '#059669', 'font-size': '1.5rem', 'margin-top': '0'}),
                            html.P(f"Enter a floating-point target for {target_col.replace('_', ' ')}.", style={'font-size': '0.9rem', 'color': '#065f46'}),
                            dcc.Input(
                                id='target-value-input',
                                type='number', # Use type='number' for float input
                                value=initial_target_value, # Initialized to initial_current_mean
                                placeholder='e.g., 5.5',
                                style={'width': '90%', 'padding': '10px', 'border': '1px solid #ccc', 'border-radius': '4px', 'margin-top': '10px'}
                            ),
                        ]
                    ),
                ]
            ),

            # --- Scorecards ---
            html.Div(
                className="scorecard-container",
                style={'display': 'flex', 'gap': '20px', 'margin-bottom': '20px'},
                children=[
                    # Scorecard 1: Current Target Value (Original Mean)
                    html.Div(
                        style={'flex': 1, 'background-color':'#f9ecd8','padding':'12px 16px','border-radius':'8px','text-align':'center', 'border': '1px solid #e5e7eb'},
                        children=[
                            html.Div(f"Current {target_col.replace('_', ' ')}", style={'font-size':'15px','font-weight':'600','margin-bottom':'2px'}),
                            html.Div("Overall (All Regions)", style={'font-size':'12px','color':'#777','margin-bottom':'6px'}),
                            html.Div(id='scorecard-current-value', style={'font-size':'24px','font-weight':'700'}, children=[f'{initial_current_mean:.2f}'])
                        ]
                    ),

                    # Scorecard 2: Target Value (New Goal Scorecard, in the middle)
                    html.Div(
                        style={'flex': 1, 'background-color':'#e6f4f1','padding':'12px 16px','border-radius':'8px','text-align':'center', 'border': '1px solid #e5e7eb'},
                        children=[
                            html.Div(f"Target Value for {target_col.replace('_', ' ')}", style={'font-size':'15px','font-weight':'600','margin-bottom':'2px', 'color': '#047857'}),
                            html.Div("Goal Set by User", style={'font-size':'12px','color':'#777','margin-bottom':'6px'}),
                            # Initialized to initial_current_mean, updated by target-value-input
                            html.Div(id='scorecard-goal-value', style={'font-size':'24px','font-weight':'700', 'color': '#047857'}, children=[f'{initial_current_mean:.2f}'])
                        ]
                    ),

                    # Scorecard 3 (Original 2): Average Change
                    html.Div(
                        style={'flex': 1, 'background-color':'#f9ecd8','padding':'12px 16px','border-radius':'8px','text-align':'center', 'border': '1px solid #e5e7eb'},
                        children=[
                            html.Div(f"Average Change in {target_col.replace('_', ' ')}", style={'font-size':'15px','font-weight':'600','margin-bottom':'2px'}),
                            html.Div("Per Selected Regions (or Overall)", style={'font-size':'12px','color':'#777','margin-bottom':'6px'}),
                            html.Div(id='scorecard-change-value', style={'font-size':'24px','font-weight':'700'}, children=['0.00'])
                        ]
                    ),
                ]
            ),
            # --- End Scorecards ---

            # --- NEW 3-Column Map and Chart Row (UPDATED TO EQUAL WIDTHS) ---
            html.Div(
                style={'display': 'flex', 'gap': '10px', 'margin-bottom': '20px'},
                children=[
                    # Map Chart (Column 1 - 33.33%)
                    html.Div(
                        style={'flex': '1', 'background-color': '#ffffff', 'padding': '5px', 'border-radius': '8px', 'border': '1px solid #e5e7eb'},
                        children=[
                            html.H3("🗺️ Geographic Impact", style={'color': '#1e3a8a', 'font-size': '1.5rem', 'margin-top': '0'}),
                            html.Div(
                                style={'height': '360px', 'overflowY': 'hidden'},
                                children=[
                                    dcc.Loading(
                                        id="loading-map",
                                        type="default",
                                        children=dcc.Graph(
                                            id='impact-map-chart',
                                            figure=initial_map_figure,
                                            config={
                                                'modeBarButtonsToAdd': ['lasso2d', 'box_select'],
                                                'displaylogo': False
                                            }
                                        )
                                    )
                                ]
                            ),
                            # SWAPPED ELEMENT 1: Clear Button (Below Map, aligned LEFT)
                            html.Div(
                                style={'display': 'flex', 'justify-content': 'flex-start', 'margin-top': '10px', 'padding-top': '10px', 'border-top': '1px solid #eee'},
                                children=[
                                    html.Button(
                                        '🗑️ Clear District Selections',
                                        id='clear-map-selection-button',
                                        n_clicks=0,
                                        style={
                                            'background-color': '#ef4444',
                                            'color': 'white',
                                            'padding': '6px 12px',
                                            'border': 'none',
                                            'border-radius': '4px',
                                            'cursor': 'pointer',
                                            'font-weight': '600'
                                        }
                                    )
                                ]
                            )
                        ]
                    ),

                    # District Bar Chart (Column 2 - 33.33%)
                    html.Div(
                        style={'flex': '1', 'background-color': '#ffffff', 'padding': '5px', 'border-radius': '8px', 'border': '1px solid #e5e7eb'},
                        children=[
                            html.H3("📈 District Impact Summary", style={'color': '#1e3a8a', 'font-size': '1.5rem', 'margin-top': '0'}),
                            html.Div(
                                style={'height': '360px', 'overflowY': 'scroll'},
                                children=[
                                    dcc.Loading(
                                        id="loading-bar",
                                        type="default",
                                        children=dcc.Graph(id='impact-bar-chart', figure=initial_bar_figure)
                                    )
                                ]
                            ),
                            # SWAPPED ELEMENT 2: Selected Districts Output (Below Bar Chart)
                            html.Div(
                                id='selected-districts-container',
                                style={'margin-top': '10px', 'padding-top': '10px', 'border-top': '1px solid #eee', 'font-size': '0.9rem', 'min-height': '30px'},
                                children=[
                                    html.Div(id='selected-districts-output', children=['Use box, or lasso tools on the map to choose regions.'])
                                ]
                            ),
                            # NEW BUTTON: See District Wise Prescriptions (TRIGGER NAVIGATION)

                        ]
                    ),

                    # NEW Feature Impact Bar Chart Panel (Column 3 - 33.33%)
                    html.Div(
                        style={'flex': '1', 'background-color': '#ffffff', 'padding': '5px', 'border-radius': '8px', 'border': '1px solid #e5e7eb'},
                        children=[
                            html.H3("📊 Prescriptions", style={'color': '#a16207', 'font-size': '1.5rem', 'margin-top': '0'}),

                            # CUSTOM LEGEND (Small font, near title)
                            html.Div(
                                # Reduced margin-bottom from 10px to 5px
                                style={'font-size': '0.75rem', 'display': 'flex', 'gap': '15px', 'margin-bottom': '5px', 'justify-content': 'center'},
                                children=[
                                    html.Div([
                                        html.Span(style={'background-color': '#1d4ed8', 'display': 'inline-block', 'width': '8px', 'height': '8px', 'margin-right': '4px', 'border-radius': '2px'}),
                                        "Original Avg"
                                    ]),
                                    html.Div([
                                        html.Span(style={'background-color': '#059669', 'display': 'inline-block', 'width': '8px', 'height': '8px', 'margin-right': '4px', 'border-radius': '2px'}),
                                        "Modified Avg"
                                    ]),
                                ]
                            ),
                            dcc.Loading(
                                id="loading-feature-impact",
                                type="default",
                                children=html.Div(
                                    id='feature-impact-panel',
                                    style={'height': '360px', 'overflowY': 'scroll'}
                                )
                            ),
                            html.Div(
                                style={'display': 'flex','justify-content': 'flex-start', 'flex-direction': 'column', 'align-items': 'center', 'margin-top': '10px'},
                                children=[
                                    html.Button(
                                        '➡️ See District Wise Prescriptions',
                                        id='open-detail-modal', # This button now triggers navigation
                                        n_clicks=0,
                                        style={
                                            'background-color': '#059669',
                                            'color': 'white',
                                            'padding': '8px 16px',
                                            'border': 'none',
                                            'border-radius': '4px',
                                            'cursor': 'pointer',
                                            'font-weight': '600',
                                            'transition': 'background-color 0.3s'
                                        }
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            # --- End 3-Column Map and Chart Row ---

            # --- Bucket Selector (Now below map/chart) ---
            html.Div(
                style={'background-color': '#ffffff', 'padding': '15px', 'border-radius': '8px', 'border': '1px solid #e5e7eb', 'margin-bottom': '20px'},
                children=[
                    html.H3("🧺 Feature Bucket Filter", style={'color': '#a16207', 'font-size': '1.5rem', 'margin-top': '0'}),
                    html.P("Select one or more buckets to filter features. Click on the bar charts in the Prescriptions panel to adjust intervention sensitivity.", style={'font-size': '0.9rem', 'color': '#713f12'}),

                    dcc.Dropdown(
                        id='bucket-selector-checklist',
                        options=[{'label': name, 'value': name} for name in bucket_names],
                        value=bucket_names,
                        multi=True,
                        clearable=True,
                        style={'margin-top': '10px', 'margin-bottom': '10px'}
                        # Removed placeholder output div: html.Div(id='prescriptions-link-output', ...)
                    ),

                    html.P(id='filtered-feature-count', style={'margin-top': '15px', 'font-weight': '600', 'color': '#a16207'})
                ]
            ),
            # --- End Bucket Selector ---

            # ------------------ DOWNLOAD SECTION ------------------
            html.Div(
                style={'background-color': '#ffffff', 'padding': '15px', 'border-radius': '8px', 'border': '1px solid #e5e7eb', 'margin-top': '20px', 'margin-bottom': '20px'},
                children=[
                    html.H3("📤 Download Prescriptions", style={'color': '#a16207', 'font-size': '1.5rem', 'margin-top': '0'}),
                    html.P("Select the format and download the complete, calculated output data.", style={'font-size': '0.9rem', 'color': '#713f12', 'margin-bottom': '10px'}),

                    # Format Radio
                    dcc.RadioItems(
                        id='download-format-radio',
                        options=[
                            {'label': 'CSV', 'value': 'csv'},
                            {'label': 'Excel', 'value': 'xlsx'}
                        ],
                        value='csv',
                        inline=True,
                        style={'margin-bottom': '15px'}
                    ),

                    # Download Button
                    html.Button(
                        '⬇️ Download Data',
                        id='download-button',
                        n_clicks=0,
                        style={
                            'background-color': '#a16207',
                            'color': 'white',
                            'padding': '8px 16px',
                            'border': 'none',
                            'border-radius': '4px',
                            'cursor': 'pointer',
                            'font-weight': '600'
                        }
                    ),
                    # Hidden Download Component
                    dcc.Download(id="download-data-file")
                ]
            ),
            # ------------------ END DOWNLOAD SECTION ------------------

            # --- Final Data Table Row (Wrapped in Loading) ---
            html.Div(
                style={'background-color': '#ffffff', 'padding': '15px', 'border-radius': '8px', 'border': '1px solid #e5e7eb'},
                children=[
                    html.H3("📋 Modified Predictions per District", style={'color': '#1e3a8a', 'font-size': '1.5rem', 'margin-top': '0'}),
                    dcc.Loading(
                        id="loading-table",
                        type="default",
                        children=dash_table.DataTable(
                            id='final-data-table',
                            columns=initial_columns,
                            data=initial_data,
                            # Styling
                            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                            style_cell={'textAlign': 'left', 'padding': '10px'},
                            style_table={'overflowX': 'auto', 'border': '1px solid #e5e7eb', 'border-radius': '8px'}
                        )
                    )
                ]
            ),
            # --- End Final Data Table Row ---

            # --- Hidden Divs/Stores for Data Storage ---
            html.Div(id='feature-buckets-data', style={'display': 'none'}, children=buckets_data_json),
            html.Div(id='app-state-data', style={'display': 'none'}, children=json.dumps(app_state)),
        ]
    )


# ==============================================================================
# 4. CORE DASHBOARD LAYOUT - INITIALIZATION (DEBUG=TRUE ADDED)
# ==============================================================================

def create_dash_app(df: pd.DataFrame, app_state: dict, geo_col: str, gdf: gpd.GeoDataFrame | None) -> Dash:
    """
    Initializes the Dash app, including multi-page routing structure and callback registration.
    """
    # --- CHANGE 1: Explicitly set debug=True for full Dash DevTools/Error handling ---
    app = Dash(__name__)

    # --- Extract initial data for layout ---
    target_col = app_state.get("target_col", "N/A")
    trained_models = app_state.get("trained_models", {})
    model_names = list(trained_models.keys())
    initial_selected_model = model_names[0] if model_names else "No Models Available"
    initial_current_mean = df[target_col].mean() if target_col in df.columns else 0.0
    initial_target_value = initial_current_mean
    feature_buckets_grouped = app_state.get("feature_buckets_grouped", {})
    sig_feats = app_state.get("sig_feats", [])
    all_grouped_feats = set(sum(feature_buckets_grouped.values(), []))
    unassigned_feats = [f for f in sig_feats if f not in all_grouped_feats]
    display_buckets_grouped = feature_buckets_grouped.copy()
    display_buckets_grouped["Unassigned"] = unassigned_feats
    bucket_names = sorted(list(display_buckets_grouped.keys()))
    buckets_data = {
        "grouped": display_buckets_grouped,
        "all_sig_feats": sig_feats,
        "original_df_json": df.to_json(orient='split')
    }
    buckets_data_json = json.dumps(buckets_data)
    initial_sens_map = {feat: 0.5 for feat in sig_feats}
    initial_df_mod = calculate_impact(df, app_state, initial_target_value, initial_sens_map, initial_selected_model)

    initial_bar_figure = generate_bar_chart(initial_df_mod, app_state)
    initial_map_figure = generate_map(initial_df_mod, app_state, gdf, selected_districts=[])

    temp_df = initial_df_mod.copy()
    temp_df.rename(columns={target_col: "Target (Original)", "Predicted": "Predicted", "Change": "Change"}, inplace=True)
    numeric_cols = temp_df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        temp_df[numeric_cols] = temp_df[numeric_cols].round(4)
    initial_data = temp_df.to_dict('records')
    # Pre-populate intervention feature columns in the initial display data for use in the detail page
    intervened_feature_cols = []
    for feat in sig_feats:
        if feat in df.columns:
            original_col_name = f"{feat} (Original)"
            modified_col_name = f"{feat} (Modified)"
            temp_df[original_col_name] = df[feat].round(4)
            # The modified value is already in temp_df[feat], so we rename it
            if feat in temp_df.columns and feat != modified_col_name:
                temp_df.rename(columns={feat: modified_col_name}, inplace=True)
            intervened_feature_cols.extend([original_col_name, modified_col_name])

    initial_columns = _prepare_data_table_columns(temp_df, geo_col, sig_feats, "Target (Original)", "Predicted", "Change")
    initial_display_data_json = temp_df.to_json(orient='records') # Now includes original/modified feature values

    effective_geo_col = geo_col
    if effective_geo_col not in initial_df_mod.columns:
        effective_geo_col = 'District' if 'District' in initial_df_mod.columns else 'N/A'

    initial_store_data_json = '[]'
    if effective_geo_col != "N/A":
        # UPDATED: Include "Predicted" column in the impact data store
        initial_store_df = initial_df_mod[[effective_geo_col, target_col, 'Predicted', 'Change']].copy()
        initial_store_df.rename(columns={target_col: "Target (Original)"}, inplace=True)
        initial_store_df['GeoUnitName'] = initial_store_df[effective_geo_col]
        # UPDATED: Store all four columns for use in the duel bar chart
        initial_store_data_json = initial_store_df[["GeoUnitName", "Target (Original)", "Predicted", "Change"]].to_json(orient='records')

    initial_feature_averages = {}
    for feat in sig_feats:
        if feat in df.columns and feat in initial_df_mod.columns:
            original_mean = df[feat].mean()
            modified_mean = initial_df_mod[feat].mean()
            # FIX: Corrected feature_averages to initial_feature_averages
            initial_feature_averages[feat] = {'original': original_mean, 'modified': modified_mean}
    initial_feature_averages_json = json.dumps(initial_feature_averages)

    # --- Main Application Layout with Router ---
    app.layout = html.Div(
        style={
            'padding': '0', # Remove padding from top level, added in page layouts
            'font-family': 'Inter, sans-serif',
            'background-color': '#f3f4f6' # Set a light gray background for the overall app
        },
        children=[
            # Hidden storage elements remain outside the routing div
            dcc.Store(id='selected-districts-store', data=[]),
            dcc.Store(id='full-impact-data-store', data=initial_store_data_json), # Contains: GeoUnitName, Target(Original), Predicted, Change (SUMMARY)
            dcc.Store(id='full-display-data-store', data=initial_display_data_json), # Contains: GeoUnitCol, Feat(Orig), Feat(Mod), Target(Orig), Predicted, Change (FULL DETAIL)
            dcc.Store(id='feature-averages-store', data=initial_feature_averages_json),
            dcc.Store(id='intervened-feature-list-store', data=sig_feats),
            dcc.Store(id='calculation-trigger-flag', data=0),
            dcc.Store(id='sensitivity-map-store', data=initial_sens_map),
            dcc.Store(id='active-feature-store', data={'feature_name': None, 'clicks': 0}),
            dcc.Store(id='map-figure-store', data=initial_map_figure.to_json()), # Cache Map figure JSON
            dcc.Store(id='bar-figure-store', data=initial_bar_figure.to_json()), # Cache Bar figure JSON
            dcc.Download(id="download-data-file"),


            # Popover (Modal) must be visible on all pages, so it stays outside the main content switch
            html.Div(
                id='sensitivity-popover-container',
                style={
                    'display': 'none',
                    'position': 'fixed',
                    'z-index': '1001',
                    'right': '40px',
                    'top': '250px',
                    'width': 'auto',
                    'height': 'auto',
                    'box-shadow': '0 4px 12px rgba(0, 0, 0, 0.15)',
                    'border-radius': '8px',
                    'overflow': 'visible',
                    'max-width': '280px',
                },
                children=[
                    html.Div(
                        id='sensitivity-popover-content',
                        style={
                            'background-color': '#e0f2fe',
                            'padding': '15px',
                            'border': '1px solid #93c5fd',
                            'border-radius': '8px',
                            'color': '#1e3a8a'
                        },
                        children=[
                            html.Div(
                                style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'border-bottom': '1px solid #93c5fd', 'padding-bottom': '8px', 'margin-bottom': '10px'},
                                children=[
                                    html.H4("Feature Sensitivity", id='modal-feature-title', style={'margin': '0', 'color': '#1d4ed8', 'font-size': '1.1rem'}),
                                    html.Button(
                                        'Apply',
                                        id='modal-close-button',
                                        n_clicks=0,
                                        style={
                                            'color': '#047857',
                                            'font-weight': 'bold',
                                            'font-size': '14px',
                                            'cursor': 'pointer',
                                            'border': '1px solid #047857',
                                            'background-color': '#e6f4f1',
                                            'padding': '4px 8px',
                                            'border-radius': '4px',
                                        }
                                    )
                                ]
                            ),
                            html.P("Current Intervention Sensitivity:", style={'margin-bottom': '5px', 'font-size': '0.9rem'}),
                            html.Div(id='modal-sensitivity-value', style={'font-size': '2.0rem', 'font-weight': '700', 'color': '#059669', 'text-align': 'center', 'padding': '5px', 'border': '1px dashed #05966966', 'border-radius': '4px', 'margin-bottom': '10px'}),
                            html.Div(id='popover-slider-container', style={'padding': '10px 0'}),
                            html.P("Adjust the value using the slider above.", style={'font-size': '0.75rem', 'color': '#1d4ed8', 'text-align': 'center', 'margin-top': '10px'})
                        ]
                    )
                ]
            ),


            # Routing components
            dcc.Location(id='url', refresh=False),
            # This Div changes content based on the URL (only handles home now)
            html.Div(id='page-content', style={'padding': '20px'}),
        ]
    )

    # --- REGISTER DETAIL PAGE CALLBACKS ---
    # Register the callbacks for the /details route
    register_detail_page_callbacks(app,app_state)
    # --------------------------------------

    # --- CALLBACK: Redirect Button to /details page ---
    @app.callback(
        Output('url', 'pathname', allow_duplicate=True),
        [Input('open-detail-modal', 'n_clicks')],
        prevent_initial_call=True
    )
    def navigate_to_detail_page(n_clicks):
        """Redirects the user to the /details page when the button is clicked."""
        if n_clicks and n_clicks > 0:
            return '/details'
        return no_update

    # --- Router Callback (Updated to call imported layout function) ---
    @app.callback(Output('page-content', 'children'),
                  [Input('url', 'pathname')],
                  [State('selected-districts-store', 'data')])
    def display_page(pathname, selected_districts):
        if pathname == '/' or pathname == '/prescriptive_dashboard.py':
            # This is the main dashboard page
            return generate_main_dashboard_layout(
                initial_current_mean, initial_target_value, initial_bar_figure,
                initial_map_figure, initial_columns, initial_data, target_col,
                model_names, initial_selected_model, bucket_names,
                buckets_data_json, app_state, initial_sens_map,
                initial_store_data_json, initial_display_data_json,
                initial_feature_averages_json, sig_feats
            )
        elif pathname == '/details':
            # This is the new detail page/dash (CALLS IMPORTED FUNCTION)
            return generate_detail_page_layout(selected_districts)
        else:
            return html.Div([
                html.H1('404: Page Not Found', style={'color': '#ef4444'}),
                html.P(f'The path "{pathname}" was not recognized.', style={'margin-bottom': '20px'}),
                html.A(
                    html.Button('⬅️ Go to Home',
                                style={'background-color': '#1d4ed8', 'color': 'white', 'padding': '8px 16px', 'border': 'none', 'border-radius': '4px', 'cursor': 'pointer', 'font-weight': '600'}),
                    href='/',
                    style={'text-decoration': 'none'}
                )
            ])

    # --- CALLBACK: Update Sensitivity Map Store (Single Source of Truth) ---
    @app.callback(
        Output('sensitivity-map-store', 'data'),
        Input({'type': 'sensitivity-slider', 'index': ALL}, 'value'),
        [State({'type': 'sensitivity-slider', 'index': ALL}, 'id'),
         State('sensitivity-map-store', 'data')],
        prevent_initial_call=True
    )
    def update_sensitivity_map_store(sens_values: List[float], sens_ids: List[Dict[str, str]], current_map: Dict[str, float]) -> Dict[str, float]:
        """
        Updates the persistent sensitivity map store whenever any dynamic slider component
        (main or popover, since they share IDs) is moved.
        """
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        new_map = current_map.copy()

        for i, id_obj in enumerate(sens_ids):
            feature_name = id_obj.get('index')
            if feature_name:
                new_map[feature_name] = sens_values[i]

        return new_map

    # --- CALLBACK: Chart Click to Store (Activates Popover) ---
    @app.callback(
        Output('active-feature-store', 'data'),
        [Input({'type': 'feature-impact-chart', 'index': ALL}, 'clickData'),
         Input({'type': 'feature-impact-chart-detail', 'index': ALL}, 'clickData')], # Input from detail page chart
        [State('active-feature-store', 'data')],
        prevent_initial_call=True
    )
    def handle_feature_chart_click(main_click_data_list: List[Any], detail_click_data_list: List[Any], current_store_data: Dict[str, Any]):
        """Detects a click on any feature bar chart (main or detail page) and stores the feature name."""
        ctx = callback_context

        if not ctx.triggered:
            return no_update

        triggered_id_str = ctx.triggered[0]['prop_id'].split('.')[0]

        try:
            triggered_id = json.loads(triggered_id_str)
            feature_name = triggered_id.get('index')
        except json.JSONDecodeError:
            return no_update

        # Check if the click event actually contains valid data (a point was clicked)
        click_data_list = main_click_data_list + detail_click_data_list
        clicked = any(cd and 'points' in cd and len(cd['points']) > 0 for cd in click_data_list if cd)

        if feature_name and clicked:
            new_clicks = current_store_data.get('clicks', 0) + 1
            return {'feature_name': feature_name, 'clicks': new_clicks}

        return no_update

    # --- CALLBACK: RESET ACTIVE FEATURE ON CLOSE BUTTON CLICK AND TRIGGER CALCULATION ---
    @app.callback(
        [Output('active-feature-store', 'data', allow_duplicate=True),
         Output('calculation-trigger-flag', 'data')],
        Input('modal-close-button', 'n_clicks'),
        State('calculation-trigger-flag', 'data'),
        prevent_initial_call=True
    )
    def reset_active_feature_on_close_and_trigger_calc(n_clicks, current_flag):
        """
        Instantly hides the modal and triggers the main calculation via the flag.
        """
        if n_clicks and n_clicks > 0:
            reset_active_data = {'feature_name': None, 'clicks': 0}
            new_flag = current_flag + 1
            return reset_active_data, new_flag
        return no_update, no_update

    # --- CALLBACK: Dynamic Slider in Popover (Updated to use sensitivity-map-store) ---
    @app.callback(
        Output('popover-slider-container', 'children'),
        [Input('active-feature-store', 'data')],
        [State('sensitivity-map-store', 'data')]
    )
    def update_popover_slider(active_feature_data: Dict[str, Any], sensitivity_map: Dict[str, float]):
        """Dynamically renders the synchronized slider inside the popover, reading value from store."""
        feature_name = active_feature_data.get('feature_name')

        if not feature_name or active_feature_data.get('clicks', 0) == 0:
            return html.Div("Select a chart to view the corresponding slider.", style={'font-size': '0.8rem', 'color': '#777'})

        current_value = sensitivity_map.get(feature_name, 0.5)

        slider_component = dcc.Slider(
            id={'type': 'sensitivity-slider', 'index': feature_name},
            min=0.0,
            max=1.0,
            step=0.05,
            value=current_value,
            marks={0: '0.0', 0.5: '0.5', 1: '1.0'},
            tooltip={"placement": "bottom", "always_visible": True}
        )
        return slider_component

    # --- CALLBACK: Show Popover and Populate Content (Updated to use sensitivity-map-store) ---
    @app.callback(
        [Output('sensitivity-popover-container', 'style'),
         Output('modal-feature-title', 'children'),
         Output('modal-sensitivity-value', 'children')],
        [Input('active-feature-store', 'data'),
         Input('sensitivity-map-store', 'data')],
    )
    def update_modal_display(active_feature_data: Dict[str, Any], sensitivity_map: Dict[str, float]):
        """Populates the popover with sensitivity data and controls its visibility."""

        active_feature = active_feature_data.get('feature_name')
        is_active = active_feature_data.get('clicks', 0) > 0 and active_feature is not None

        popover_style = {
            'position': 'fixed',
            'z-index': '1001',
            'right': '40px',
            'top': '250px',
            'width': 'auto',
            'height': 'auto',
            'box-shadow': '0 4px 12px rgba(0, 0, 0, 0.15)',
            'border-radius': '8px',
            'overflow': 'visible',
            'max-width': '280px',
        }

        if not is_active:
            popover_style['display'] = 'none'
            return popover_style, "Feature Sensitivity", "N/A"

        sensitivity_val = sensitivity_map.get(active_feature, 0.5)

        formatted_title = f"Sensitivity for: {active_feature.replace('_', ' ').title()}"
        formatted_value = f"{sensitivity_val:.2f}"

        popover_style['display'] = 'block'
        return (
            popover_style,
            formatted_title,
            formatted_value
        )

    # --- CALLBACK: Update Goal Value Scorecard based on Input (Unchanged) ---
    @app.callback(
        Output('scorecard-goal-value', 'children'),
        [Input('target-value-input', 'value')]
    )
    def update_scorecard_goal_value(target_value_input):
        """Updates the Target Value scorecard display based on user input."""
        if target_value_input is None or target_value_input == '':
            return "N/A"
        try:
            val = float(target_value_input)
            return f'{val:.2f}'
        except ValueError:
            return "Invalid Input"

    # --- CALLBACK: Clear Map Selection (Unchanged) ---
    @app.callback(
        Output('impact-map-chart', 'selectedData'),
        [Input('clear-map-selection-button', 'n_clicks')]
    )
    def clear_map_selection(n_clicks):
        """Clears the visual selection on the map."""
        if n_clicks > 0:
            return None
        return no_update

    @app.callback(
        Output('selected-model-output', 'children'),
        [Input('model-selector-dropdown', 'value')]
    )
    def update_selected_model_display(selected_model_name):
        """Displays the name of the currently selected model."""
        return f"Currently selected model: {selected_model_name}"

    # --- CALLBACK: Filter Features by Bucket (Updated to output filtered list to store) ---
    @app.callback(
        [Output('filtered-feature-count', 'children'),
         Output('intervened-feature-list-store', 'data')],
        [Input('bucket-selector-checklist', 'value')],
        [State('feature-buckets-data', 'children')]
    )
    def filter_features_by_bucket(selected_buckets: list, buckets_data_json: str):
        """Filters the significant features based on the selected buckets and stores the list for intervention."""
        if not buckets_data_json:
            return "Error: Feature data not loaded.", []

        buckets_data = json.loads(buckets_data_json)
        feature_buckets_grouped = buckets_data["grouped"]
        sig_feats = buckets_data["all_sig_feats"]

        filtered_feats = []

        if not selected_buckets or len(selected_buckets) == len(feature_buckets_grouped):
            filtered_feats = sig_feats
        else:
            for bucket in selected_buckets:
                features = feature_buckets_grouped.get(bucket, [])
                filtered_feats.extend(features)

        filtered_feats = list(set(filtered_feats))
        filtered_feats = [f for f in filtered_feats if f in sig_feats]

        if not filtered_feats:
            count_output = "No features found after filtering."
        else:
            feature_list_str = ", ".join(filtered_feats)
            count_output = html.Div([
                f"Number of filtered features: {len(filtered_feats)}",
                html.P(f"Filtered Features: {feature_list_str}",
                       style={'font-size': '0.8rem', 'margin-top': '5px', 'word-break': 'break-all'})
            ])

        return count_output, filtered_feats

    # --- CALLBACK: Update Feature Impact Panel (Dual Bar Charts) ---
    @app.callback(
        Output('feature-impact-panel', 'children'),
        [Input('intervened-feature-list-store', 'data'),
         Input('feature-averages-store', 'data')]
    )
    def update_feature_impact_panel(selected_feats: list, feature_averages_json: str):
        """
        Generates the dual bar charts for Original vs. Modified feature averages.
        """
        if not selected_feats:
            return html.P("Select features and set sensitivity to view average impact.", style={'color': '#4a5568', 'margin': '10px'})

        try:
            feature_averages = json.loads(feature_averages_json) if feature_averages_json else {}
        except json.JSONDecodeError:
            return html.P("Error loading feature average data.", style={'color': '#ef4444', 'margin': '10px'})

        chart_elements = []

        for feat in selected_feats:
            avg_data = feature_averages.get(feat, {'original': 0.0, 'modified': 0.0})
            old_val = avg_data['original']
            new_val = avg_data['modified']

            plain_text_label = feat.replace('_', ' ').title()

            max_abs_val = max(abs(old_val), abs(new_val))
            x_range_max = max(1.0, max_abs_val * 1.1)
            min_val = min(old_val, new_val)
            x_range_min = min(-1.0, min_val * 1.1)
            x_range = [x_range_min, x_range_max]

            bar_fig = go.Figure(data=[
                go.Bar(
                    name='Original Avg',
                    x=[old_val],
                    y=[""],
                    orientation='h',
                    marker_color='#1d4ed8',
                    text=[f"{old_val:.4f}"],
                    textposition='inside',
                    insidetextanchor='middle',
                    hovertemplate='Original Average: %{x:.4f}<extra></extra>',
                    showlegend=False
                ),
                go.Bar(
                    name='Modified Avg',
                    x=[new_val],
                    y=[""],
                    orientation='h',
                    marker_color='#059669',
                    text=[f"{new_val:.4f}"],
                    textposition='inside',
                    insidetextanchor='middle',
                    hovertemplate='Modified Average: %{x:.4f}<extra></extra>',
                    showlegend=False
                )
            ])

            bar_fig.update_layout(
                barmode='group',
                height=100,
                margin=dict(l=10, r=10, t=10, b=20),
                title=None,
                xaxis=dict(title="", showgrid=True, range=x_range, showticklabels=True, tickfont=dict(size=9)),
                yaxis=dict(showticklabels=False),
                plot_bgcolor='white', paper_bgcolor='white',
                uirevision=feat
            )
            bar_fig.update_traces(
                textfont_size=9,
                cliponaxis=False
            )

            chart_elements.append(
                html.Div(
                    id={'type': 'feature-panel-div', 'index': feat},
                    style={'margin-bottom': '10px', 'padding': '0', 'border-bottom': '1px dotted #e5e7eb', 'cursor': 'pointer'},
                    children=[
                        html.P(plain_text_label, style={'font-size': '0.9rem', 'font-weight': '600', 'margin-bottom': '2px', 'margin-top': '2px', 'color': '#2d3748', 'text-align': 'center'}),
                        dcc.Graph(
                            id={'type': 'feature-impact-chart', 'index': feat},
                            figure=bar_fig,
                            config={'displayModeBar': False},
                            style={'height': '100px', 'width': '100%'}
                        )
                    ]
                )
            )

        return chart_elements

    # --- MAIN CALLBACK FOR DATA CALCULATION AND VISUALIZATION ---
    @app.callback(
        [Output('impact-map-chart', 'figure'),
         Output('final-data-table', 'columns'),
         Output('full-impact-data-store', 'data'),
         Output('full-display-data-store', 'data'),
         Output('feature-averages-store', 'data'),
         Output('map-figure-store', 'data')], # ADDED: Cache map figure JSON
        [Input('model-selector-dropdown', 'value'),
         Input('target-value-input', 'value'),
         Input('intervened-feature-list-store', 'data'),
         Input('calculation-trigger-flag', 'data'),
         Input('clear-map-selection-button', 'n_clicks'),
         Input('selected-districts-store', 'data')],
        [State('feature-buckets-data', 'children'),
         State('app-state-data', 'children'),
         State('sensitivity-map-store', 'data')]
    )
    def update_impact_table(selected_model_name, target_value_input, intervened_feats_list, calc_trigger_flag, clear_n_clicks, selected_districts, buckets_data_json, app_state_json, sensitivities_map):
        """
        Gathers all inputs, performs the calculation (prescriptive), and prepares all outputs.
        """
        intervene_feats = intervened_feats_list
        sensitivities = {k: v for k, v in sensitivities_map.items() if k in intervene_feats}
        target_value = float(target_value_input) if target_value_input is not None and target_value_input != '' else 0.0

        buckets_data = json.loads(buckets_data_json)
        app_state = json.loads(app_state_json)
        df = pd.read_json(buckets_data["original_df_json"], orient='split')
        geo_unit_col = app_state.get("geo_unit_col", "N/A")
        target_col = app_state.get("target_col", "N/A")

        gdf = None
        gdf_ref = app_state.get('gdf_ref')
        if gdf_ref and gdf_ref in PYTHON_OBJECT_LOOKUP:
            try:
                gdf = PYTHON_OBJECT_LOOKUP[gdf_ref]
            except Exception as e:
                print(f"Error retrieving GDF from lookup: {e}")

        empty_map = go.Figure(layout=dict(title=None, height=500))
        empty_map_json = empty_map.to_json()
        empty_columns = []
        empty_store_data = '[]'
        empty_display_data = '[]'
        empty_averages_data = '{}'
        is_data_invalid = (geo_unit_col == "N/A" or target_col == "N/A" or df.empty)

        if is_data_invalid:
            return empty_map, empty_columns, empty_store_data, empty_display_data, empty_averages_data, empty_map_json

        df_mod = calculate_impact(df, app_state, target_value, sensitivities, selected_model_name)

        effective_geo_col = geo_unit_col
        if geo_unit_col not in df_mod.columns:
            if 'District' in df_mod.columns:
                geo_unit_col = 'District'
            else:
                is_data_invalid = True

        if is_data_invalid:
            return empty_map, empty_columns, empty_store_data, empty_display_data, empty_averages_data, empty_map_json

        # Calculate Averages (using filtering logic)
        effective_geo_col = geo_unit_col
        is_filtered = selected_districts and len(selected_districts) > 0 and isinstance(selected_districts[0], str)
        if is_filtered and effective_geo_col in df.columns and effective_geo_col in df_mod.columns:
            df_filtered = df[df[effective_geo_col].isin(selected_districts)]
            df_mod_filtered = df_mod[df_mod[effective_geo_col].isin(selected_districts)]
        else:
            df_filtered = df
            df_mod_filtered = df_mod

        feature_averages = {}
        for feat in intervened_feats_list:
            if feat in df_filtered.columns:
                original_mean = df_filtered[feat].mean()
                modified_mean = df_mod_filtered[feat].mean()
                feature_averages[feat] = {'original': original_mean, 'modified': modified_mean}
        feature_averages_json = json.dumps(feature_averages)

        # Prepare Data Stores and Table
        # Store Data (Summary data for duel bar chart)
        store_df = df_mod[[geo_unit_col, target_col, 'Predicted', 'Change']].copy()
        store_df.rename(columns={target_col: "Target (Original)"}, inplace=True)
        store_df['GeoUnitName'] = store_df[geo_unit_col]
        store_data = store_df[["GeoUnitName", "Target (Original)", "Predicted", "Change"]].to_json(orient='records')

        # Display Data (Full data for table and detail page)
        display_df = df_mod.copy()
        intervention_cols_order = []
        original_target_col_name = target_col
        new_target_col_name = "Target (Original)"
        new_predicted_col_name = "Predicted"
        new_change_col_name = "Change"

        if original_target_col_name in display_df.columns:
            display_df.rename(columns={original_target_col_name: new_target_col_name}, inplace=True)
        if "Predicted" in display_df.columns:
            display_df.rename(columns={"Predicted": new_predicted_col_name}, inplace=True)
        if "Change" in display_df.columns:
            display_df.rename(columns={"Change": new_change_col_name}, inplace=True)

        target_col_for_summary = new_target_col_name

        if intervene_feats:
            for feat in intervene_feats:
                if feat in df.columns:
                    before_col_name_revised = f"{feat} (Original)"
                    modified_col_name_revised = f"{feat} (Modified)"
                    display_df[before_col_name_revised] = df[feat]
                    # Rename the modified column
                    if feat in display_df.columns and feat != modified_col_name_revised:
                        display_df.rename(columns={feat: modified_col_name_revised}, inplace=True)
                    intervention_cols_order.extend([before_col_name_revised, modified_col_name_revised])
                # Ensure modified column name exists even if feature not in df.columns (shouldn't happen, but defensive)
                elif f"{feat} (Modified)" not in display_df.columns:
                    if feat in display_df.columns:
                        display_df.rename(columns={feat: f"{feat} (Modified)"}, inplace=True)


        identifier_col = [col for col in [geo_unit_col] if col in display_df.columns]
        summary_result_cols = [col for col in [target_col_for_summary, new_predicted_col_name, new_change_col_name] if col in display_df.columns]

        # Ensure only unique, existing columns are selected in the correct order
        ordered_cols = identifier_col[:] + intervention_cols_order + [
            col for col in display_df.columns.tolist()
            if col not in identifier_col and col not in intervention_cols_order and col not in summary_result_cols
        ] + summary_result_cols

        final_cols = []
        seen = set()
        for col in ordered_cols:
            if col in display_df.columns and col not in seen:
                final_cols.append(col)
                seen.add(col)

        display_df = display_df[final_cols]


        numeric_cols = display_df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            display_df[numeric_cols] = display_df[numeric_cols].round(4)

        columns = _prepare_data_table_columns(display_df, geo_unit_col, intervene_feats, target_col_for_summary, new_predicted_col_name, new_change_col_name)
        full_display_data = display_df.to_json(orient='records') # Updated to include all necessary columns

        chart_app_state = app_state.copy()
        chart_app_state['geo_unit_col'] = geo_unit_col
        map_figure = generate_map(df_mod, chart_app_state, gdf, selected_districts)

        # IMPORTANT: Cache the map figure (including visual state like zoom/pan) as JSON
        # so the detail page can load the exact current view.
        map_figure_json = map_figure.to_json()

        return (
            map_figure,
            columns,
            store_data,
            full_display_data,
            feature_averages_json,
            map_figure_json
        )

    # --- CALLBACK: Dynamic Bar Chart based on Map Selection (Updated to cache bar figure) ---
    @app.callback(
        [Output('impact-bar-chart', 'figure'),
         Output('bar-figure-store', 'data')], # ADDED: Cache bar figure JSON
        [Input('selected-districts-store', 'data'),
         Input('full-impact-data-store', 'data')],
        [State('app-state-data', 'children')],
        # Prevent initial call for this callback as it relies on the page-content to exist
        prevent_initial_call='initial_only'
    )
    def update_dynamic_bar_chart(selected_districts, impact_data_json, app_state_json):
        """Filters the full impact data (Change) based on map selection and generates the bar chart."""
        empty_bar = px.bar(title=None, height=500)
        empty_bar_json = empty_bar.to_json()

        if not impact_data_json or impact_data_json == '[]':
            return empty_bar, empty_bar_json

        try:
            impact_df = pd.read_json(impact_data_json, orient='records')
            app_state = json.loads(app_state_json)
        except ValueError:
            return empty_bar, empty_bar_json

        if impact_df.empty:
            return empty_bar, empty_bar_json

        geo_key_expected = app_state.get("geo_unit_col", "N/A")
        chart_df = impact_df[['GeoUnitName', 'Change']].copy()
        if geo_key_expected not in chart_df.columns and 'District' in chart_df.columns:
            geo_key_expected = 'District'

        chart_df.rename(columns={'GeoUnitName': geo_key_expected}, inplace=True)

        if selected_districts and len(selected_districts) > 0 and isinstance(selected_districts[0], str):
            filtered_df = chart_df[chart_df[geo_key_expected].isin(selected_districts)]
        else:
            filtered_df = chart_df

        chart_app_state = app_state.copy()
        chart_app_state['geo_unit_col'] = geo_key_expected

        bar_figure = generate_bar_chart(filtered_df, chart_app_state)

        # IMPORTANT: Cache the bar figure as JSON so the detail page can load the exact current view.
        bar_figure_json = bar_figure.to_json()

        return bar_figure, bar_figure_json


    # --- CALLBACK: Dynamic Scorecards based on Map Selection (Unchanged) ---
    @app.callback(
        [Output('scorecard-current-value', 'children'),
         Output('scorecard-change-value', 'children')],
        [Input('selected-districts-store', 'data'),
         Input('full-impact-data-store', 'data')],
        prevent_initial_call='initial_only'
    )
    def update_scorecards_from_selection(selected_districts, impact_data_json):
        """Calculates and updates scorecards based on map selection."""
        if not impact_data_json or impact_data_json == '[]':
            return "N/A", "N/A"

        try:
            impact_df = pd.read_json(impact_data_json, orient='records')
        except ValueError:
            return "N/A", "N/A"

        if impact_df.empty:
            return "N/A", "N/A"

        current_mean = impact_df["Target (Original)"].mean()

        if selected_districts and len(selected_districts) > 0 and isinstance(selected_districts[0], str):
            filtered_df = impact_df[impact_df['GeoUnitName'].isin(selected_districts)]
            change_mean = filtered_df["Change"].mean() if not filtered_df.empty else np.nan
        else:
            change_mean = impact_df["Change"].mean()

        if np.isnan(current_mean) or np.isnan(change_mean):
            return "N/A", "N/A"

        return f'{current_mean:.2f}', f'{change_mean:+.2f}'


    # --- CALLBACK: Dynamic Final Data Table based on Map Selection (Unchanged) ---
    @app.callback(
        Output('final-data-table', 'data'),
        [Input('selected-districts-store', 'data'),
         Input('full-display-data-store', 'data')],
        [State('app-state-data', 'children')],
        prevent_initial_call='initial_only'
    )
    def update_final_table_data(selected_districts, display_data_json, app_state_json):
        """Filters the final data table based on the districts selected on the map."""
        if not display_data_json or display_data_json == '[]':
            return []

        app_state = json.loads(app_state_json)
        geo_unit_col = app_state.get("geo_unit_col", "N/A")

        full_df = pd.read_json(display_data_json, orient='records')

        effective_geo_col = geo_unit_col
        if effective_geo_col not in full_df.columns and 'District' in full_df.columns:
            effective_geo_col = 'District'

        if effective_geo_col not in full_df.columns:
            return full_df.to_dict('records')

        if selected_districts and len(selected_districts) > 0 and isinstance(selected_districts[0], str):
            filtered_df = full_df[full_df[effective_geo_col].isin(selected_districts)]
            return filtered_df.to_dict('records')
        else:
            return full_df.to_dict('records')


    # --- CALLBACK: Map Interactivity (Unchanged) ---
    @app.callback(
        [Output('selected-districts-store', 'data', allow_duplicate=True),
         Output('selected-districts-output', 'children', allow_duplicate=True)],
        [Input('impact-map-chart', 'selectedData')],
        [State('app-state-data', 'children')],
        prevent_initial_call='initial_only'
    )
    def update_selected_districts_list(selectedData, app_state_json):
        """Manages the list of selected districts based on map box/lasso selections."""
        app_state = json.loads(app_state_json)
        local_geo_key = app_state.get("geo_unit_col", "N/A")

        selected_districts = []

        if selectedData and 'points' in selectedData:
            for point in selectedData['points']:
                district_name = point.get('hovertext')

                if not district_name:
                    text_label = point.get('text')
                    if text_label and isinstance(text_label, str):
                        district_name = text_label.split('(')[0].strip()

                if district_name and district_name not in selected_districts:
                    selected_districts.append(district_name)

        if selected_districts:
            selected_list_content = html.P(
                [
                    html.Strong(f"Selected {local_geo_key.replace('_', ' ').title()}s ({len(selected_districts)}): "),
                    html.Span(f"{', '.join(selected_districts)}", title=", ".join(selected_districts), style={'font-weight': 'normal'})
                ],
                style={'margin': 0}
            )
        else:
            selected_list_content = "Use box, or lasso tools on the map to choose regions."

        return selected_districts, selected_list_content

    # --- CALLBACK: Download Data (Unchanged) ---
    @app.callback(
        Output("download-data-file", "data", allow_duplicate=True),
        [Input("download-button", "n_clicks")],
        [State('download-format-radio', 'value'),
         State('full-display-data-store', 'data')],
        prevent_initial_call=True
    )
    def download_data(n_clicks, format_choice, display_data_json):
        """Generates the file (CSV or Excel) for download when the button is clicked."""
        if n_clicks == 0 or not display_data_json or display_data_json == '[]':
            return no_update

        df = pd.read_json(display_data_json, orient='records')

        if format_choice == 'csv':
            csv_string_io = io.StringIO()
            df.to_csv(csv_string_io, index=False)
            csv_string_io.seek(0)
            return dcc.send_string(
                csv_string_io.read(),
                "prescriptive_modelling_output.csv",
                mime="text/csv"
            )

        elif format_choice == 'xlsx':
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name="Prescriptions")
            excel_buffer.seek(0)
            return dcc.send_bytes(
                excel_buffer.read(),
                "prescriptive_modelling_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        return no_update

    return app

# ==============================================================================
# 5. SERVER THREAD TARGET FUNCTION (use_reloader=True ADDED)
# ==============================================================================

def run_dash_server_thread_target(df_json: str, app_state_json: str, geo_col: str, port: int):
    """
    The entry point for the Dash server thread.
    It retrieves the trained models and scaler from the global lookup and adds them
    to the app_state to be accessible by the callbacks.
    """
    # 1. Deserialize JSON data passed from Streamlit thread
    df = pd.read_json(df_json, orient='split')
    app_state = json.loads(app_state_json)


    # 3. Retrieve GeoDataFrame separately (as it's used in create_dash_app)
    gdf = None
    gdf_ref = app_state.get('gdf_ref')
    if gdf_ref and gdf_ref in PYTHON_OBJECT_LOOKUP:
        try:
            gdf = PYTHON_OBJECT_LOOKUP[gdf_ref]
            # Ensure the GEO_COL is correctly set in app_state if necessary
            app_state['GEO_COL'] = app_state.get('GEO_COL', gdf.index.name)
        except Exception as e:
            print(f"Error retrieving GDF during initialization: {e}")

    # 4. Create the Dash application
    dash_app = create_dash_app(df, app_state, geo_col, gdf)

    # 5. Run the server
    run_simple(
        hostname='0.0.0.0',
        port=port,
        application=dash_app.server,
        threaded=True,
        # --- CHANGE 2: Set use_reloader=True for auto-reloading during development ---
        use_reloader=False,
        use_debugger=True
    )
