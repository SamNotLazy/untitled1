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
from typing import Dict, Any
from dash.dash_table import Format

# ==============================================================================
# 1. SHARED GLOBAL STATE AND UTILITIES
# ==============================================================================

# GLOBAL LOOKUP TABLE: This dictionary serves as the shared memory space
# where non-serializable Python objects (like trained models, scalers,
# and GeoDataFrames) are stored by Streamlit (main thread) and retrieved
# by Dash (server thread).
PYTHON_OBJECT_LOOKUP = {}

def find_free_port(default_port=8050):
    """Finds an open port starting from a default."""
    port = default_port
    while port < default_port + 50:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
            port += 1
    raise IOError("No free ports found.")

# ==============================================================================
# 2. CORE CALCULATION LOGIC (Using the actual trained model)
# ==============================================================================

def calculate_impact(
        df: pd.DataFrame,
        app_state: Dict[str, Any],
        pct_changes: Dict[str, float],
        sensitivities: Dict[str, float],
        selected_model_name: str
) -> pd.DataFrame:
    """
    Performs the differential impact analysis using the actual trained model and scaler
    retrieved from the shared memory (PYTHON_OBJECT_LOOKUP). This replicates the
    model prediction logic from the original Streamlit application.

    Args:
        df: The base DataFrame containing original feature values and the target column.
        app_state: A dictionary mimicking st.session_state (must contain the
                   references for models, scaler, and features).
        pct_changes: Dictionary mapping features to their percentage change intervention (e.g., {'feat_A': 0.1}).
        sensitivities: Dictionary mapping features to their sensitivity (0.0 to 1.0).
        selected_model_name: The name of the selected model (e.g., 'linear', 'rf').

    Returns:
        A modified DataFrame including the original target, baseline prediction, and calculated 'Change'.
    """
    df_mod = df.copy()
    target_col = app_state.get("target_col", "N/A")
    sig_feats = app_state.get("sig_feats", [])

    # --- 1. Retrieve Model, Scaler, and Feature Order from shared state ---

    # app_state contains the ID strings for non-serializable objects. We use PYTHON_OBJECT_LOOKUP
    # to retrieve the actual objects using those IDs.
    trained_models_refs = app_state.get("trained_models", {})
    scaler_id = app_state.get("scaler")
    feature_order = app_state.get("trained_feature_names", sig_feats) # The features used for training
    # Retrieve objects from the global lookup using their unique IDs
    scaler = PYTHON_OBJECT_LOOKUP.get(scaler_id)
    model_id = trained_models_refs.get(selected_model_name)
    model = PYTHON_OBJECT_LOOKUP.get(model_id)

    if model is None or not feature_order:
        # Fallback if the necessary components for real prediction are missing
        df["Predicted"] = df[target_col] if target_col in df.columns else 0.0
        df_mod["Predicted"] = df["Predicted"]
        df_mod["Change"] = 0.0
        if target_col in df.columns:
            df_mod[target_col] = df[target_col]
        return df_mod

    # Prepare base dataframes for model input
    df_to_predict = df[feature_order]
    df_mod_to_predict = df[feature_order].copy() # Start with a copy of original feature values


    # --- 2. Apply feature interventions to df_mod_to_predict ---
    for feat, pct in pct_changes.items():
        if feat in df_mod_to_predict.columns:
            sens = sensitivities.get(feat, 0.5)
            # Apply intervention: df_mod[feat] = df_mod[feat] * (1 + sens * pct)
            df_mod_to_predict[feat] = df[feat] * (1 + sens * pct)


    # --- 3. Perform Prediction (Matching Streamlit Logic) ---

    if selected_model_name == "linear":
        # Check if the scaler object is valid (i.e., not None, not a string, and has the .transform method)
        if not hasattr(scaler, 'transform'):
            # Handle case where linear model is selected but scaler is missing or invalid
            print("WARNING: Linear model selected but scaler object is missing or invalid. Skipping prediction.")
            print(target_col,df)

            df["Predicted"] = df[target_col] if target_col in df.columns else 0.0
            df_mod["Predicted"] = df["Predicted"]
        else:
            # Apply scaling
            df_scaled = scaler.transform(df_to_predict)
            df_mod_scaled = scaler.transform(df_mod_to_predict)

            # Predict
            df["Predicted"] = model.predict(df_scaled)
            df_mod["Predicted"] = model.predict(df_mod_scaled)
    else:
        # For tree-based or other models that don't need scaling
        df["Predicted"] = model.predict(df_to_predict)
        df_mod["Predicted"] = model.predict(df_mod_to_predict)

    # --- 4. Calculate Raw Change and Apply Direction Masking ---

    # Change is the difference between modified prediction and baseline prediction
    df_mod["Change"] = df_mod["Predicted"] - df["Predicted"]

    if pct_changes:
        # Access masking parameters
        direction = app_state.get("target_direction", "Increase")
        positive_indicators = app_state.get("final_positive", [])
        negative_indicators = app_state.get("final_negative", [])

        # --- Apply the complex Direction Masking logic (MATCHING THE STREAMLIT FILE) ---
        for feat in pct_changes:
            pct = pct_changes[feat]

            if direction == "Increase":
                # Goal: Maximize Change. Mask if intervention *contradicts* expected change.
                if feat in positive_indicators:
                    # Positive factor: Positive PCT should result in POSITIVE Change.
                    df_mod["Change"] = df_mod["Change"].mask((pct > 0) & (df_mod["Change"] < 0), 0)
                    # Negative PCT should result in NEGATIVE Change.
                    df_mod["Change"] = df_mod["Change"].mask((pct < 0) & (df_mod["Change"] > 0), 0)
                elif feat in negative_indicators:
                    # Negative factor: Positive PCT should result in NEGATIVE Change.
                    df_mod["Change"] = df_mod["Change"].mask((pct > 0) & (df_mod["Change"] > 0), 0)
                    # Negative PCT should result in POSITIVE Change.
                    df_mod["Change"] = df_mod["Change"].mask((pct < 0) & (df_mod["Change"] < 0), 0)

            elif direction == "Decrease":
                # Goal: Minimize Change. Mask if intervention *contradicts* expected change.
                if feat in positive_indicators:
                    # Positive factor: Positive PCT should result in POSITIVE Change (BAD for decrease). Mask if Change is POSITIVE.
                    df_mod["Change"] = df_mod["Change"].mask((pct > 0) & (df_mod["Change"] > 0), 0)
                    # Negative PCT should result in NEGATIVE Change (GOOD for decrease). Mask if Change is NEGATIVE.
                    df_mod["Change"] = df_mod["Change"].mask((pct < 0) & (df_mod["Change"] < 0), 0)
                elif feat in negative_indicators:
                    # Negative factor: Positive PCT should result in NEGATIVE Change (GOOD for decrease). Mask if Change is NEGATIVE.
                    df_mod["Change"] = df_mod["Change"].mask((pct > 0) & (df_mod["Change"] < 0), 0)
                    # Negative PCT should result in POSITIVE Change (BAD for decrease). Mask if Change is POSITIVE.
                    df_mod["Change"] = df_mod["Change"].mask((pct < 0) & (df_mod["Change"] > 0), 0)

    # Store the original target for display purposes and merge intervened features back
    if target_col in df.columns:
        df_mod[target_col] = df[target_col]

    for feat in pct_changes:
        if feat in df_mod.columns:
            # We update the original df_mod with the intervened feature value
            df_mod[feat] = df_mod_to_predict[feat]

    return df_mod
def generate_bar_chart(change_df: pd.DataFrame, app_state: dict) -> px.bar:
    """
    Generates the Bar Chart for impact visualization.
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
        empty_bar = px.bar(title=None, height=500)
        return empty_bar

    # Filter out null values
    bar_df = plot_df[plot_df["Change"].notnull()].copy()

    if bar_df.empty:
        empty_bar = px.bar(title=None, height=500)
        return empty_bar

    # --- Bar Chart Logic ---
    bar_df = bar_df.sort_values(bar_col, ascending=True).copy()

    # Calculate dynamic height to ensure a fixed bar thickness of approximately 20px per bar.
    N_districts = len(bar_df[geo_key].unique())
    dynamic_height = max(450, N_districts * 25 + 100)

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
        if 'District' in plot_df.columns:
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
    minx, miny, maxx, maxy = gdf.total_bounds
    center_lon = (minx + maxx) / 2
    center_lat = (miny + maxy) / 2

    width = abs(maxx - minx)
    height = abs(maxy - miny)
    max_dim = max(width, height)

    initial_zoom = 5.0
    if max_dim > 0:
        calculated_zoom = np.log2(360 / max_dim) - 0.5
        initial_zoom = max(5, min(9, calculated_zoom))

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
        center={"lat": center_lat, "lon": center_lon},
        zoom=initial_zoom
    )

    # --- Add Scattermapbox trace for labels ---
    map_fig.add_trace(go.Scattermapbox(
        lat=plot_df['centroid_lat'],
        lon=plot_df['centroid_lon'],
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
        selected_indices = plot_df[plot_df[geo_key_for_select].isin(selected_districts)].index.tolist()

        if selected_indices:
            map_fig.data[0].selectedpoints = selected_indices
            map_fig.update_layout(selectionrevision=True)
    # --- END SELECTION PRESERVATION LOGIC ---

    map_fig.update_mapboxes(
        bearing=0,
        pitch=0,
        center={"lat": center_lat, "lon": center_lon},
        accesstoken=None, # Clear access token
        layers=[],
        zoom=initial_zoom
    )

    map_fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title=None,
        height=fixed_height
    )

    return map_fig


# ==============================================================================
# 3. DASH APPLICATION CORE
# ==============================================================================

def create_dash_app(df: pd.DataFrame, app_state: dict, geo_col: str, gdf: gpd.GeoDataFrame | None) -> Dash:
    """
    Initializes the Dash app using the data retrieved from the Streamlit thread.
    """
    app = Dash(__name__)

    # Extract core variables
    target_col = app_state.get("target_col", "N/A")

    # --- Prepare Model Data for Dropdown ---
    trained_models = app_state.get("trained_models", {})
    model_names = list(trained_models.keys())
    # Retrieve the actual object references for initial calculation

    # We need to map model IDs back to model names for the dropdown options.
    # Since trained_models is a dict of {name: ID}, we can't easily get the object names
    # unless we iterate over the actual objects, but here we only need the names.
    model_names = list(trained_models.keys())
    initial_selected_model = model_names[0] if model_names else "No Models Available"

    # We must ensure the initial calculation uses the correct, retrieved objects.
    # Since the object retrieval is now inside calculate_impact, we can proceed.

    # --- Prepare Feature Data for Filter and Sliders ---
    feature_buckets_grouped = app_state.get("feature_buckets_grouped", {})
    sig_feats = app_state.get("sig_feats", [])

    # Dynamic bucket preparation logic: Add "Unassigned" features
    all_grouped_feats = set(sum(feature_buckets_grouped.values(), []))
    unassigned_feats = [f for f in sig_feats if f not in all_grouped_feats]

    display_buckets_grouped = feature_buckets_grouped.copy()
    display_buckets_grouped["Unassigned"] = unassigned_feats

    bucket_names = list(display_buckets_grouped.keys())
    bucket_names.sort()

    # Store feature data in a hidden Div
    buckets_data = {
        "grouped": display_buckets_grouped,
        "all_sig_feats": sig_feats,
        "original_df_json": df.to_json(orient='split') # Store original DF for calculation
    }
    buckets_data_json = json.dumps(buckets_data)


    # --- 0. Initialize Calculation and Chart for Default Display ---
    # Perform initial calculation (zero intervention) to get baseline predictions
    initial_df_mod = calculate_impact(df, app_state, {}, {}, initial_selected_model)

    # Calculate initial scorecard values
    initial_current_mean = df[target_col].mean() if target_col in df.columns else 0.0

    # Generate the initial map and bar chart figures
    initial_bar_figure = generate_bar_chart(initial_df_mod, app_state)
    initial_map_figure = generate_map(initial_df_mod, app_state, gdf, selected_districts=[])

    # Apply initial renaming/rounding for default table view consistency
    temp_df = initial_df_mod.copy()
    temp_df.rename(columns={
        target_col: "Target (Original)",
        "Predicted": "Predicted",
        "Change": "Change"
    }, inplace=True)
    numeric_cols = temp_df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        temp_df[numeric_cols] = temp_df[numeric_cols].round(4)

    # Initialize the final data table with the baseline data
    initial_data = temp_df.to_dict('records')

    # Prepare initial columns with explicit numeric formatting
    initial_columns = []
    summary_column_names = ["Change", "Predicted", "Target (Original)"]

    for i in temp_df.columns:
        col_def = {"name": i, "id": i}
        if temp_df[i].dtype in [np.float64, np.float32, np.float16] or i in summary_column_names:
            col_def['type'] = 'numeric'
            col_def['format'] = Format.Format(precision=4, scheme=Format.Scheme.fixed)
        initial_columns.append(col_def)

    # Store initial full formatted data
    initial_display_data_json = temp_df.to_json(orient='records')

    # Initialize the full impact data store
    effective_geo_col = geo_col
    if effective_geo_col not in initial_df_mod.columns:
        if 'District' in initial_df_mod.columns:
            effective_geo_col = 'District'
        else:
            effective_geo_col = 'N/A'

    if effective_geo_col == "N/A":
        initial_store_data_json = '[]'
    else:
        initial_store_df = initial_df_mod[[effective_geo_col, target_col, 'Change']].copy()
        initial_store_df.rename(columns={target_col: "Target (Original)"}, inplace=True)
        initial_store_df['GeoUnitName'] = initial_store_df[effective_geo_col]
        initial_store_data_json = initial_store_df[["GeoUnitName", "Target (Original)", "Change"]].to_json(orient='records')


    app.layout = html.Div(
        style={
            'padding': '20px',
            'font-family': 'Inter, sans-serif',
            'background-color': '#ffffff'
        },
        children=[
            html.H1("Dash Impact Analysis Dashboard", style={'color': '#1e3a8a', 'border-bottom': '2px solid #60a5fa', 'padding-bottom': '10px'}),

            # --- Input Controls Row ---
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

                    # Bucket Selector
                    html.Div(
                        style={'background-color': '#ffffff', 'padding': '15px', 'border-radius': '8px', 'border': '1px solid #e5e7eb'},
                        children=[
                            html.H3("🧺 Feature Bucket Filter", style={'color': '#a16207', 'font-size': '1.5rem', 'margin-top': '0'}),
                            html.P("Select one or more buckets to filter features.", style={'font-size': '0.9rem', 'color': '#713f12'}),

                            dcc.Dropdown(
                                id='bucket-selector-checklist',
                                options=[{'label': name, 'value': name} for name in bucket_names],
                                value=bucket_names,
                                multi=True,
                                clearable=True,
                                style={'margin-top': '10px', 'margin-bottom': '10px'}
                            ),

                            html.P(id='filtered-feature-count', style={'margin-top': '15px', 'font-weight': '600', 'color': '#a16207'})
                        ]
                    ),
                ]
            ),

            # --- Intervention Controls ---
            html.Div(
                style={'background-color': '#ffffff', 'padding': '15px', 'border-radius': '8px', 'margin-bottom': '20px', 'border': '1px solid #e5e7eb'},
                children=[
                    html.H3("🛠️ Define Feature Interventions", style={'color': '#2f855a', 'font-size': '1.5rem', 'margin-top': '0'}),

                    # 1. Feature Multi-select for Intervention
                    html.Label("📊 Choose Features to Intervene On", style={'font-weight': '600', 'margin-top': '10px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='intervene-feats-selector',
                        options=[{'label': f, 'value': f} for f in sig_feats],
                        value=[],  # Default to no features selected
                        multi=True,
                        clearable=True,
                        style={'margin-bottom': '15px'}
                    ),

                    # 2. Dynamic Sliders Container
                    html.Div(id='feature-sliders-container', children=[
                        html.P("Select features above to define intervention % change and sensitivity.", style={'color': '#4a5568'})
                    ])
                ]
            ),

            # --- Scorecards ---
            html.Div(
                className="scorecard-container",
                style={'display': 'flex', 'gap': '20px', 'margin-bottom': '20px'},
                children=[
                    # Scorecard 1: Current Target Value
                    html.Div(
                        style={'flex': 1, 'background-color':'#f9ecd8','padding':'12px 16px','border-radius':'8px','text-align':'center', 'border': '1px solid #e5e7eb'},
                        children=[
                            html.Div(f"Current {target_col.replace('_', ' ')}", style={'font-size':'15px','font-weight':'600','margin-bottom':'2px'}),
                            html.Div("Overall (All Regions)", style={'font-size':'12px','color':'#777','margin-bottom':'6px'}),
                            html.Div(id='scorecard-current-value', style={'font-size':'24px','font-weight':'700'}, children=[f'{initial_current_mean:.2f}'])
                        ]
                    ),
                    # Scorecard 2: Average Change
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

            # --- Map and Chart Row ---
            html.Div(
                style={'display': 'flex', 'gap': '5px', 'margin-bottom': '20px'},
                children=[
                    # Map Chart (Left Column)
                    html.Div(
                        style={'flex': 1, 'background-color': '#ffffff', 'padding': '10px', 'border-radius': '8px', 'border': '1px solid #e5e7eb'},
                        children=[
                            html.H3("🗺️ Geographic Impact", style={'color': '#1e3a8a', 'font-size': '1.5rem', 'margin-top': '0'}),
                            html.Div(
                                style={'height': '400px', 'overflowY': 'hidden'},
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
                            # Selected Districts Output (Below Map)
                            html.Div(
                                id='selected-districts-container',
                                style={'margin-top': '10px', 'padding-top': '10px', 'border-top': '1px solid #eee', 'font-size': '0.9rem', 'min-height': '30px'},
                                children=[
                                    html.Div(id='selected-districts-output', children=['Use box, or lasso tools on the map to choose regions.'])
                                ]
                            )
                        ]
                    )
                    ,
                    # Bar Chart (Right Column)
                    html.Div(
                        style={'flex': 1, 'background-color': '#ffffff', 'padding': '10px', 'border-radius': '8px', 'border': '1px solid #e5e7eb'},
                        children=[
                            html.H3("📈 District Impact Summary", style={'color': '#1e3a8a', 'font-size': '1.5rem', 'margin-top': '0'}),
                            html.Div(
                                style={'height': '400px', 'overflowY': 'scroll'},
                                children=[
                                    dcc.Loading(
                                        id="loading-bar",
                                        type="default",
                                        children=dcc.Graph(id='impact-bar-chart', figure=initial_bar_figure)
                                    )
                                ]
                            ),
                            # Clear Button (Below Bar Chart, aligned LEFT)
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
                    )
                ]
            ),
            # --- End Map and Chart Row ---

            # --- Final Data Table Row ---
            html.Div(
                style={'background-color': '#ffffff', 'padding': '15px', 'border-radius': '8px', 'border': '1px solid #e5e7eb', 'margin-top': '20px'},
                children=[
                    html.H3("📋 Modified Predictions per District", style={'color': '#1e3a8a', 'font-size': '1.5rem', 'margin-top': '0'}),

                    # --- NEW: Download Buttons ---
                    html.Div(
                        style={'display': 'flex', 'gap': '10px', 'margin-bottom': '15px', 'justify-content': 'flex-end'},
                        children=[
                            html.Button(
                                '⬇️ Download CSV',
                                id='btn-download-csv',
                                n_clicks=0,
                                style={
                                    'background-color': '#22c55e',
                                    'color': 'white',
                                    'padding': '6px 12px',
                                    'border': 'none',
                                    'border-radius': '4px',
                                    'cursor': 'pointer',
                                    'font-weight': '600',
                                    'transition': 'background-color 0.3s'
                                }
                            ),
                            html.Button(
                                '⬇️ Download Excel',
                                id='btn-download-excel',
                                n_clicks=0,
                                style={
                                    'background-color': '#1d4ed8',
                                    'color': 'white',
                                    'padding': '6px 12px',
                                    'border': 'none',
                                    'border-radius': '4px',
                                    'cursor': 'pointer',
                                    'font-weight': '600',
                                    'transition': 'background-color 0.3s'
                                }
                            )
                        ]
                    ),
                    # --- END NEW: Download Buttons ---

                    dash_table.DataTable(
                        id='final-data-table',
                        columns=initial_columns,
                        data=initial_data,
                        # Styling
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                        style_cell={'textAlign': 'left', 'padding': '10px'},
                        style_table={'overflowX': 'auto', 'border': '1px solid #e5e7eb', 'border-radius': '8px'}
                    ),
                ]
            ),
            # --- End Final Data Table Row ---

            # --- Hidden Divs/Stores for Data Storage ---
            html.Div(id='feature-buckets-data', style={'display': 'none'}, children=buckets_data_json),
            html.Div(id='app-state-data', style={'display': 'none'}, children=json.dumps(app_state)),
            # Store for selected districts (used to hold state from selectedData)
            dcc.Store(id='selected-districts-store', data=[]),
            # Store for the full calculated impact data (GeoUnitName, Target (Original), Change)
            dcc.Store(id='full-impact-data-store', data=initial_store_data_json),
            # Store for the full formatted data table (used as source for filtering the final table)
            dcc.Store(id='full-display-data-store', data=initial_display_data_json),
            # --- NEW: Download Component ---
            dcc.Download(id="download-data-file"),
        ]
    )

    # --- Dash Callbacks ---

    # --- CALLBACK: Clear Map Selection ---
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

    @app.callback(
        [Output('filtered-feature-count', 'children'),
         Output('intervene-feats-selector', 'options'), # Update intervention options
         Output('intervene-feats-selector', 'value')], # Clear intervention selection
        [Input('bucket-selector-checklist', 'value')],
        [State('feature-buckets-data', 'children'),
         State('intervene-feats-selector', 'value')] # Keep current selection attempt
    )
    def filter_features_by_bucket(selected_buckets: list, buckets_data_json: str, current_intervene_feats: list):
        """Filters the significant features based on the selected buckets and updates the intervention options."""
        if not buckets_data_json:
            return "Error: Feature data not loaded.", [], []

        buckets_data = json.loads(buckets_data_json)
        feature_buckets_grouped = buckets_data["grouped"]
        sig_feats = buckets_data["all_sig_feats"]

        filtered_feats = []

        # Logic: If no buckets selected, default to all significant features
        if not selected_buckets or len(selected_buckets) == len(feature_buckets_grouped):
            filtered_feats = sig_feats
        else:
            for bucket in selected_buckets:
                features = feature_buckets_grouped.get(bucket, [])
                filtered_feats.extend(features)

        filtered_feats = list(set(filtered_feats))
        filtered_feats = [f for f in filtered_feats if f in sig_feats]

        # Prepare options for the intervention dropdown
        intervention_options = [{'label': f, 'value': f} for f in filtered_feats]

        # Keep only the previously selected features that are still in the filtered list
        new_intervene_selection = [f for f in current_intervene_feats if f in filtered_feats]

        if not filtered_feats:
            count_output = "No features found after filtering."
        else:
            feature_list_str = ", ".join(filtered_feats)
            count_output = html.Div([
                f"Number of filtered features: {len(filtered_feats)}",
                html.P(f"Filtered Features: {feature_list_str}",
                       style={'font-size': '0.8rem', 'margin-top': '5px', 'word-break': 'break-all'})
            ])

        return count_output, intervention_options, new_intervene_selection

    @app.callback(
        Output('feature-sliders-container', 'children'),
        [Input('intervene-feats-selector', 'value')]
    )
    def update_feature_sliders(selected_feats: list):
        """Dynamically generates % Change and Sensitivity sliders for selected features."""
        if not selected_feats:
            return html.P("Select features above to define intervention % change and sensitivity.", style={'color': '#4a5568'})

        slider_elements = []

        for feat in selected_feats:
            # Define default values for the sliders
            default_pct = 0.0
            default_sens = 0.5

            slider_elements.append(
                html.Div(
                    id=f'slider-group-{feat}',
                    style={'border': '1px solid #d1d5db', 'border-radius': '6px', 'padding': '15px', 'margin-top': '10px', 'background-color': '#ffffff'},
                    children=[
                        html.H4(f"🔧 {feat}", style={'font-size': '1.2rem', 'margin-top': '0', 'color': '#2d3748'}),

                        # % Change Slider
                        html.Div(
                            style={'margin-bottom': '20px'},
                            children=[
                                html.Label("📉 % Change (-100% to +100%)", style={'font-weight': '500', 'display': 'block', 'margin-bottom': '10px'}),
                                dcc.Slider(
                                    # NOTE: Using Pattern Matching IDs for dynamic inputs
                                    id={'type': 'pct-change-slider', 'index': feat},
                                    min=-100,
                                    max=100,
                                    step=1,
                                    value=default_pct,
                                    marks={i: {'label': f'{i}%'} for i in [-100, -50, 0, 50, 100]},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ]
                        ),

                        # Sensitivity Slider
                        html.Div(
                            children=[
                                html.Label("🎯 Sensitivity (0.0 - 1.0)", style={'font-weight': '500', 'display': 'block', 'margin-bottom': '10px'}),
                                dcc.Slider(
                                    # NOTE: Using Pattern Matching IDs for dynamic inputs
                                    id={'type': 'sensitivity-slider', 'index': feat},
                                    min=0.0,
                                    max=1.0,
                                    step=0.05,
                                    value=default_sens,
                                    marks={0: '0.0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1.0'},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ]
                        ),
                    ]
                )
            )

        return slider_elements

    # --- MAIN CALLBACK FOR DATA CALCULATION AND VISUALIZATION ---
    @app.callback(
        [Output('impact-map-chart', 'figure'),
         Output('final-data-table', 'columns'),
         Output('full-impact-data-store', 'data'),
         Output('full-display-data-store', 'data')],
        [Input('model-selector-dropdown', 'value'),
         Input('intervene-feats-selector', 'value'),
         Input({'type': 'pct-change-slider', 'index': ALL}, 'value'),
         Input({'type': 'sensitivity-slider', 'index': ALL}, 'value'),
         # FIX: Added clear button input to force map figure regeneration
         Input('clear-map-selection-button', 'n_clicks'),
         # FIX: Made selected-districts an Input to ensure map redraws instantly after clear store update
         Input('selected-districts-store', 'data')],
        [State('feature-buckets-data', 'children'),
         State('app-state-data', 'children'),
         State({'type': 'pct-change-slider', 'index': ALL}, 'id'),
         State({'type': 'sensitivity-slider', 'index': ALL}, 'id')]
    )
    def update_impact_table(selected_model_name, intervene_feats, pct_values, sens_values, n_clicks, selected_districts, buckets_data_json, app_state_json, pct_ids, sens_ids):
        """
        Gathers all inputs, performs the calculation (placeholder), and prepares all outputs.
        """
        # 1. Reconstruct pct_changes and sensitivities Dictionaries
        pct_changes = {pct_ids[i]['index']: v / 100.0 for i, v in enumerate(pct_values)}
        sensitivities = {sens_ids[i]['index']: v for i, v in enumerate(sens_values)}

        # Filter to only include features currently selected for intervention
        pct_changes = {k: v for k, v in pct_changes.items() if k in intervene_feats}
        sensitivities = {k: v for k, v in sensitivities.items() if k in intervene_feats}

        # 2. Retrieve Data
        buckets_data = json.loads(buckets_data_json)
        app_state = json.loads(app_state_json)
        df = pd.read_json(buckets_data["original_df_json"], orient='split')
        geo_unit_col = app_state.get("geo_unit_col", "N/A")
        target_col = app_state.get("target_col", "N/A")

        # Retrieve GeoDataFrame from global lookup
        gdf = None
        gdf_ref = app_state.get('gdf_ref')
        if gdf_ref and gdf_ref in PYTHON_OBJECT_LOOKUP:
            try:
                gdf = PYTHON_OBJECT_LOOKUP[gdf_ref]
            except Exception as e:
                print(f"Error retrieving GDF from lookup: {e}")



        # Placeholder figures/data for invalid state
        empty_map = go.Figure(layout=dict(title=None, height=500))
        empty_columns = []
        empty_store_data = '[]'
        empty_display_data = '[]'

        is_data_invalid = (geo_unit_col == "N/A" or target_col == "N/A" or df.empty)
        if is_data_invalid:
            return empty_map, empty_columns, empty_store_data, empty_display_data

        # 3. Perform Calculation (using REAL model)
        df_mod = calculate_impact(df, app_state, pct_changes, sensitivities, selected_model_name)

        # Robust Geo Column Check - Must run AFTER df_mod is created
        original_geo_unit_col = geo_unit_col
        if geo_unit_col not in df_mod.columns:
            if 'District' in df_mod.columns:
                geo_unit_col = 'District'
            else:
                is_data_invalid = True

        if is_data_invalid:
            return empty_map, empty_columns, empty_store_data, empty_display_data

        # 4. Data for Stores and Table Preparation

        # --- Data for the full-impact-data-store (for scorecards) ---
        store_df = df_mod[[geo_unit_col, target_col, 'Change']].copy()
        store_df.rename(columns={
            target_col: "Target (Original)",
        }, inplace=True)
        store_df['GeoUnitName'] = store_df[geo_unit_col]
        store_data = store_df[["GeoUnitName", "Target (Original)", "Change"]].to_json(orient='records')
        # --- End Data for Scorecard Store ---


        # --- Data Table Preparation (full display data for filtering) ---
        display_df = df_mod.copy()

        intervention_cols_order = []

        # 0. Rename the summary/target columns for concise display
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
            # We already applied intervention to df_mod's features inside calculate_impact
            for feat in intervene_feats:
                if feat in df.columns:
                    before_col_name_revised = f"{feat} (Original)"
                    modified_col_name_revised = f"{feat} (Modified)"

                    display_df[before_col_name_revised] = df[feat]

                    # The feature column itself in display_df now holds the modified value
                    if feat in display_df.columns and feat != modified_col_name_revised:
                        display_df.rename(columns={feat: modified_col_name_revised}, inplace=True)

                    intervention_cols_order.extend([before_col_name_revised, modified_col_name_revised])

        # --- Final Column Ordering and Selection ---
        identifier_col = [col for col in [geo_unit_col] if col in display_df.columns]
        summary_result_cols = [col for col in [target_col_for_summary, new_predicted_col_name, new_change_col_name] if col in display_df.columns]

        final_cols = identifier_col[:]
        final_cols.extend(intervention_cols_order)
        all_display_cols = display_df.columns.tolist()

        for col in all_display_cols:
            is_core_summary = col in summary_result_cols
            is_identifier = col in identifier_col

            if not is_identifier and not is_core_summary and col not in final_cols:
                final_cols.append(col)

        for col in summary_result_cols:
            if col in display_df.columns:
                final_cols.append(col)

        display_df = display_df[final_cols]

        # APPLY ROUNDING to all numeric columns to 4 decimal places
        numeric_cols = display_df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            display_df[numeric_cols] = display_df[numeric_cols].round(4)

        # Prepare columns for Dash DataTable with explicit formatting for precision
        columns = []
        summary_column_names_list = [new_change_col_name, new_predicted_col_name, target_col_for_summary]

        for i in display_df.columns:
            col_def = {"name": i, "id": i}
            if display_df[i].dtype in [np.float64, np.float32, np.float16] or i in summary_column_names_list:
                col_def['type'] = 'numeric'
                col_def['format'] = Format.Format(precision=4, scheme=Format.Scheme.fixed)
            columns.append(col_def)

        # Store the full, formatted data as JSON for the filtering step
        full_display_data = display_df.to_json(orient='records')
        # --- End Data Table Preparation ---


        # 5. Chart Generation
        chart_app_state = app_state.copy()
        chart_app_state['geo_unit_col'] = geo_unit_col

        map_figure = generate_map(df_mod, chart_app_state, gdf, selected_districts)

        return (
            map_figure,
            columns,
            store_data,
            full_display_data
        )

    # --- CALLBACK: Dynamic Bar Chart based on Map Selection ---
    @app.callback(
        Output('impact-bar-chart', 'figure'),
        [Input('selected-districts-store', 'data'), # List of selected districts from map
         Input('full-impact-data-store', 'data')], # Full calculated data
        [State('app-state-data', 'children')]
    )
    def update_dynamic_bar_chart(selected_districts, impact_data_json, app_state_json):
        """Filters the full impact data (Change) based on map selection and generates the bar chart."""
        empty_bar = px.bar(title=None, height=500)

        if not impact_data_json or impact_data_json == '[]':
            return empty_bar

        try:
            impact_df = pd.read_json(impact_data_json, orient='records')
            app_state = json.loads(app_state_json)
        except ValueError:
            return empty_bar

        if impact_df.empty:
            return empty_bar

        geo_key_expected = app_state.get("geo_unit_col", "N/A")
        chart_df = impact_df[['GeoUnitName', 'Change']].copy()
        if geo_key_expected not in chart_df.columns and 'District' in chart_df.columns:
            geo_key_expected = 'District'

        chart_df.rename(columns={'GeoUnitName': geo_key_expected}, inplace=True)

        # Filtering logic
        if selected_districts and len(selected_districts) > 0 and isinstance(selected_districts[0], str):
            filtered_df = chart_df[chart_df[geo_key_expected].isin(selected_districts)]
        else:
            filtered_df = chart_df

        chart_app_state = app_state.copy()
        chart_app_state['geo_unit_col'] = geo_key_expected

        return generate_bar_chart(filtered_df, chart_app_state)


    # --- CALLBACK: Dynamic Scorecards based on Map Selection ---
    @app.callback(
        [Output('scorecard-current-value', 'children'),
         Output('scorecard-change-value', 'children')],
        [Input('selected-districts-store', 'data'),
         Input('full-impact-data-store', 'data')]
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

        # 1. Current Mean (Always overall)
        current_mean = impact_df["Target (Original)"].mean()

        # 2. Change Mean (Dynamic based on selection)
        if selected_districts and len(selected_districts) > 0 and isinstance(selected_districts[0], str):
            filtered_df = impact_df[impact_df['GeoUnitName'].isin(selected_districts)]
            change_mean = filtered_df["Change"].mean() if not filtered_df.empty else np.nan
        else:
            change_mean = impact_df["Change"].mean()

        if np.isnan(current_mean) or np.isnan(change_mean):
            return "N/A", "N/A"

        return f'{current_mean:.2f}', f'{change_mean:.2f}'


    # --- CALLBACK: Dynamic Final Data Table based on Map Selection ---
    @app.callback(
        Output('final-data-table', 'data'),
        [Input('selected-districts-store', 'data'),
         Input('full-display-data-store', 'data')],
        [State('app-state-data', 'children')]
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

        # Check if any districts are selected
        if selected_districts and len(selected_districts) > 0 and isinstance(selected_districts[0], str):
            filtered_df = full_df[full_df[effective_geo_col].isin(selected_districts)]
            return filtered_df.to_dict('records')
        else:
            return full_df.to_dict('records')


    # --- CALLBACK: Map Interactivity ---
    @app.callback(
        [Output('selected-districts-store', 'data'),
         Output('selected-districts-output', 'children')],
        [Input('impact-map-chart', 'selectedData')],
        [State('app-state-data', 'children')]
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

        # Update the output text display
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

    # --- NEW: CALLBACK FOR DOWNLOADING DATA TABLE ---
    @app.callback(
        Output("download-data-file", "data"),
        [Input("btn-download-csv", "n_clicks"),
         Input("btn-download-excel", "n_clicks")],
        [State('final-data-table', 'data'),
         State('final-data-table', 'columns')]
    )
    def download_data(n_clicks_csv, n_clicks_excel, table_data, table_columns):
        """Triggers the download of the currently filtered data table in the selected format."""
        ctx = callback_context

        if not ctx.triggered or not table_data or (n_clicks_csv is None and n_clicks_excel is None):
            return no_update

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Guard against initial callback firing with no data
        if not table_data:
            return no_update

        # Create DataFrame from the currently displayed table data
        df_to_export = pd.DataFrame(table_data)

        # Ensure correct column order/selection based on the visible columns
        if table_columns:
            column_ids = [col['id'] for col in table_columns]
            df_to_export = df_to_export[[col for col in column_ids if col in df_to_export.columns]]

        # Determine format and filename
        if button_id == 'btn-download-csv':
            # Use dcc.send_data_frame with to_csv
            return dcc.send_data_frame(
                df_to_export.to_csv,
                filename="impact_analysis_data.csv",
                index=False,
            )
        elif button_id == 'btn-download-excel':
            # Use dcc.send_data_frame with to_excel
            return dcc.send_data_frame(
                df_to_export.to_excel,
                filename="impact_analysis_data.xlsx",
                index=False,
                sheet_name="ImpactData"
            )

        return no_update
    # --- END NEW: CALLBACK FOR DOWNLOADING DATA TABLE ---


    return app

# ==============================================================================
# 4. SERVER THREAD TARGET FUNCTION
# ==============================================================================

def run_dash_server_thread_target(df_json: str, app_state_json: str, geo_col: str, port: int):
    """
    The entry point for the Dash server thread.
    It retrieves the trained models and scaler from the global lookup and adds them
    to the app_state to be accessible by the callbacks.
    """
    try:
        # 1. Deserialize JSON data passed from Streamlit thread
        df = pd.read_json(df_json, orient='split')
        app_state = json.loads(app_state_json)




        # 3. Retrieve GeoDataFrame separately (as it's used in create_dash_app)
        gdf = None
        gdf_ref = app_state.get('gdf_ref')
        if gdf_ref and gdf_ref in PYTHON_OBJECT_LOOKUP:
            try:
                gdf = PYTHON_OBJECT_LOOKUP[gdf_ref]
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
            use_reloader=False
        )

    except Exception as e:
        print(f"FATAL ERROR: Dash Server Thread failed to start or run: {e}")
        PYTHON_OBJECT_LOOKUP['thread_error'] = str(e)
