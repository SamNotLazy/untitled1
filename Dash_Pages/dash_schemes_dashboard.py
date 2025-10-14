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
from typing import Dict, Any, Tuple
from dash.dash_table import Format
import io

# ==============================================================================
# 1. SHARED GLOBAL STATE AND UTILITIES
# ==============================================================================

# GLOBAL LOOKUP TABLE: This dictionary serves as the shared memory space
# where non-serializable Python objects (like GeoDataFrames) are stored.
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
# 2. CORE CALCULATION LOGIC (Budget Allocation)
# ==============================================================================

def compute_budget_allocation(
        df: pd.DataFrame,
        model_importances: Dict[str, float],
        total_budget: float,
        geo_unit_col: str,
        mode: str = "inverse"
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Placeholder function to compute feature budgets and district allocations.
    It currently uses proportional allocation based on the absolute importance.
    The 'mode' argument is ignored for this placeholder but kept for signature consistency.
    """
    # 1. Calculate overall importance sum
    total_importance = sum(abs(imp) for imp in model_importances.values()) or 1

    # 2. Calculate feature budgets (proportional to importance)
    feature_budgets = {}
    for feature, importance in model_importances.items():
        # Using absolute importance for proportional allocation
        budget = (abs(importance) / total_importance) * total_budget
        feature_budgets[feature] = budget

    # 3. Calculate district allocations (proportional to feature values)
    district_allocations = df[[geo_unit_col]].drop_duplicates().set_index(geo_unit_col)
    district_allocations["Total_Budget"] = 0.0
    district_allocations_df = df[[geo_unit_col]].drop_duplicates().copy()

    for feature, budget in feature_budgets.items():
        if feature in df.columns:
            factor_values = df[[geo_unit_col, feature]].dropna(subset=[feature])
            factor_sum = factor_values[feature].sum()

            if factor_sum > 0:
                # Calculate proportion per district for this feature
                factor_values[f"{feature}_Allocation"] = (factor_values[feature] / factor_sum) * budget
                # Merge into the district_allocations_df table
                alloc_to_merge = factor_values.set_index(geo_unit_col)[f"{feature}_Allocation"].rename(feature)
                district_allocations_df = district_allocations_df.merge(alloc_to_merge, left_on=geo_unit_col, right_index=True, how='left')
                district_allocations_df[feature] = district_allocations_df[feature].fillna(0.0)
            else:
                district_allocations_df[feature] = 0.0
        else:
            district_allocations_df[feature] = 0.0

    # Sum up total allocation for each district
    alloc_cols = [col for col in district_allocations_df.columns if col != geo_unit_col]
    district_allocations_df["Total_Budget"] = district_allocations_df[alloc_cols].sum(axis=1)

    return feature_budgets, district_allocations_df

# ==============================================================================
# 3. VISUALIZATION UTILITIES (Adapted from previous code)
# ==============================================================================

def generate_bar_chart(change_df: pd.DataFrame, geo_key: str, bar_title: str) -> px.bar:
    """Generates the Bar Chart for budget allocation visualization."""
    plot_df = change_df.copy()
    bar_col = "Change" # 'Allocated_Budget' renamed to 'Change'

    if geo_key not in plot_df.columns or plot_df.empty:
        return px.bar(title=None, height=500)

    # Filter out null values
    bar_df = plot_df[plot_df["Change"].notnull()].copy()

    if bar_df.empty:
        return px.bar(title=None, height=500)

    # Use RdYlGn_r color scale as specified in Streamlit code
    color_scale = "RdYlGn_r"

    bar_df = bar_df.sort_values(bar_col, ascending=True).copy()

    # Calculate dynamic height
    N_districts = len(bar_df[geo_key].unique())
    dynamic_height = max(450, N_districts * 25 + 100)

    bar_fig = px.bar(
        bar_df,
        x=bar_col, y=geo_key, orientation='h',
        title=bar_title,
        labels={geo_key: 'District', bar_col: 'Budget Allocation (Cr)'},
        color=bar_col,
        color_continuous_scale=color_scale,
        height=dynamic_height,
        text=bar_df[bar_col].round(2),
        hover_data={
            geo_key: False,
            bar_col: ':.2f'
        }
    )
    bar_fig.update_traces(
        texttemplate="₹ %{text} Cr",
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=10, color="black"),
        cliponaxis=False
    )
    bar_fig.update_coloraxes(showscale=False)

    bar_fig.update_layout(
        margin=dict(l=150, r=0, t=50, b=0),
        bargap=0.1,
        yaxis=dict(tickfont=dict(size=10)),
        uirevision=True # Keep bar chart state consistent
    )

    return bar_fig

def generate_map(change_df: pd.DataFrame, gdf: gpd.GeoDataFrame | None, geo_key: str, GEO_COL: str, map_title: str, selected_districts: list | None = None) -> px.choropleth_mapbox:
    """Generates the Plotly Mapbox Choropleth map for budget allocation."""
    plot_df = change_df.copy()
    map_col = "Change"
    selected_districts = selected_districts or []

    # Use RdYlGn_r color scale as specified in Streamlit code
    color_scale = "RdYlGn_r"
    fixed_height = 500

    if gdf is None or gdf.empty or geo_key not in plot_df.columns:
        return go.Figure(layout=dict(title=map_title, height=fixed_height, margin=dict(l=0, r=0, t=20, b=0)))

    # --- Merge shapefile with change data ---
    plot_df = gdf.merge(change_df[[geo_key, map_col]], left_on=GEO_COL, right_on=geo_key, how="left").copy()
    plot_df = plot_df[plot_df[map_col].notnull()]

    if plot_df.empty:
        return go.Figure(layout=dict(title=map_title, height=fixed_height, margin=dict(l=0, r=0, t=20, b=0)))

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
    except Exception:
        plot_df['centroid_lon'] = center_lon
        plot_df['centroid_lat'] = center_lat


    # --- Choropleth Map ---
    map_fig = px.choropleth_mapbox(
        plot_df,
        geojson=plot_df.__geo_interface__,
        locations=plot_df.index,
        color=map_col,
        hover_name=geo_key,
        hover_data={
            map_col: ':.2f',
            geo_key: False,
        },
        mapbox_style="white-bg",
        opacity=0.7,
        color_continuous_scale=color_scale,
        labels={map_col: 'Budget (Cr)'},
        center={"lat": center_lat, "lon": center_lon},
        zoom=initial_zoom
    )

    # --- Add Scattermapbox trace for labels ---
    map_fig.add_trace(go.Scattermapbox(
        lat=plot_df['centroid_lat'],
        lon=plot_df['centroid_lon'],
        mode='text',
        text=plot_df[geo_key] + " (₹" + plot_df[map_col].round(2).astype(str) + " Cr)",
        textfont=dict(size=10, color="black"),
        hoverinfo='skip',
        name='Labels',
        showlegend=False
    ))

    # --- SELECTION PRESERVATION LOGIC ---
    if selected_districts and plot_df is not None and not plot_df.empty and map_fig.data:
        selected_indices = plot_df[plot_df[geo_key].isin(selected_districts)].index.tolist()
        if selected_indices:
            map_fig.data[0].selectedpoints = selected_indices
            map_fig.update_layout(selectionrevision=True)

    map_fig.update_mapboxes(
        bearing=0,
        pitch=0,
        center={"lat": center_lat, "lon": center_lon},
        accesstoken=None,
        layers=[],
        zoom=initial_zoom
    )

    map_fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title=map_title,
        height=fixed_height
    )

    return map_fig

# ==============================================================================
# 4. DASH APPLICATION CORE
# ==============================================================================

def create_dash_app(df_final: pd.DataFrame, app_state: dict, geo_col: str, gdf: gpd.GeoDataFrame | None) -> Dash:
    """Initializes the Dash app for Budget Allocation Schemes."""
    app = Dash(__name__)

    # --- Core Data Extraction ---
    target_col = app_state.get("target_col", "N/A")
    feature_importances = app_state.get("feature_importances", {})
    feature_schemes = app_state.get("feature_schemes", {})
    feature_buckets_grouped = app_state.get("feature_buckets_grouped", {})
    total_budget = app_state.get("total_budget", 1000)
    allocation_mode = app_state.get("allocation_mode", "inverse")
    GEO_COL = app_state.get("GEO_COL", "N/A")

    model_names = list(feature_importances.keys())
    initial_selected_model = model_names[0] if model_names else "No Models Available"

    # --- Prepare Bucket and Feature Data ---
    sig_feats = df_final.columns.drop([geo_col, target_col], errors='ignore').tolist()
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
        "original_df_json": df_final.to_json(orient='split'),
        "feature_importances": feature_importances,
        "feature_schemes": feature_schemes,
        "total_budget": total_budget,
        "allocation_mode": allocation_mode,
        "geo_unit_col": geo_col,
        "GEO_COL": GEO_COL
    }
    buckets_data_json = json.dumps(buckets_data)

    # --- Initial Calculation ---
    initial_importances_raw = feature_importances.get(initial_selected_model, {})
    if isinstance(initial_importances_raw, pd.DataFrame):
        initial_importances = dict(zip(initial_importances_raw["Feature"], initial_importances_raw["Importance"]))
    else:
        initial_importances = initial_importances_raw

    # Compute budget for initial model
    initial_feature_budgets, initial_district_allocations = compute_budget_allocation(
        df_final, initial_importances, total_budget, geo_col, allocation_mode
    )

    # Filter out features with no budget for scheme selection
    available_factors_initial = [f for f, b in initial_feature_budgets.items() if b is not None and b > 0]
    initial_scheme_options = {}
    for factor in available_factors_initial:
        initial_scheme_options[factor] = feature_schemes.get(factor, factor)

    initial_scheme_name = list(initial_scheme_options.values())[0] if initial_scheme_options else "No Schemes"

    # Initial scheme-specific allocation
    initial_factor_normalized = next((f for f, s in initial_scheme_options.items() if s == initial_scheme_name), None)
    initial_scheme_budget = initial_feature_budgets.get(initial_factor_normalized, 0.0)

    # Prepare data for map/bar (using FULL data for coloring)
    initial_change_df = pd.DataFrame()
    if initial_factor_normalized and initial_factor_normalized in initial_district_allocations.columns:
        initial_change_df = initial_district_allocations[[geo_col, initial_factor_normalized]].copy()
        initial_change_df.rename(columns={initial_factor_normalized: "Change"}, inplace=True)
        # Use FULL allocation DF for the total budget scorecard
        initial_total_allocated = initial_district_allocations["Total_Budget"].sum()
    else:
        initial_total_allocated = 0.0

    # Initial figures
    initial_bar_figure = generate_bar_chart(initial_change_df, geo_col, "District-wise Scheme Budget")
    initial_map_figure = generate_map(initial_change_df, gdf, geo_col, GEO_COL, "Geographic Distribution of Scheme Budget", selected_districts=[])

    # Initial scheme table data (using UNFILTERED data for consistency before selection)
    initial_scheme_table_df = pd.DataFrame()
    if not initial_change_df.empty:
        initial_scheme_table_df = initial_change_df[[geo_col, "Change"]].copy()
        initial_scheme_table_df.columns = [geo_col, f"{initial_scheme_name} Budget"]
        initial_scheme_table_df['GeoUnitName'] = initial_scheme_table_df[geo_col]

    initial_scheme_table_data = initial_scheme_table_df.to_dict('records')
    initial_scheme_table_columns = [
        {"name": geo_col, "id": geo_col},
        {"name": f"{initial_scheme_name} Budget", "id": f"{initial_scheme_name} Budget", 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)}
    ]

    # Initial full allocation table data
    initial_full_alloc_df = initial_district_allocations.copy()
    rename_map = {col: feature_schemes.get(col, col) for col in initial_full_alloc_df.columns if col not in [geo_col, "Total_Budget"]}
    initial_full_alloc_df.rename(columns=rename_map, inplace=True)
    initial_full_alloc_data = initial_full_alloc_df.to_dict('records')


    app.layout = html.Div(
        style={'padding': '20px', 'font-family': 'Inter, sans-serif', 'background-color': '#f8fafc'},
        children=[
            html.H1("📂 Budget Allocation - Schemes Dashboard", style={'color': '#0d9488', 'border-bottom': '2px solid #2dd4bf', 'padding-bottom': '10px'}),
            html.P(f"Allocation Mode: {allocation_mode}", style={'font-style': 'italic', 'color': '#0f766e'}),

            # --- Model Selector Row ---
            html.Div(
                style={'display': 'grid', 'grid-template-columns': 'repeat(auto-fit, minmax(300px, 1fr))', 'gap': '20px', 'margin-bottom': '20px'},
                children=[
                    # Model Selector
                    html.Div(
                        style={'background-color': '#ffffff', 'padding': '15px', 'border-radius': '8px', 'border': '1px solid #e5e7eb', 'box-shadow': '0 4px 6px rgba(0,0,0,0.05)'},
                        children=[
                            html.H3("🧠 Select Model", style={'color': '#0e7490', 'font-size': '1.5rem', 'margin-top': '0'}),
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
                        style={'background-color': '#ffffff', 'padding': '15px', 'border-radius': '8px', 'border': '1px solid #e5e7eb', 'box-shadow': '0 4px 6px rgba(0,0,0,0.05)'},
                        children=[
                            html.H3("🧺 Feature Bucket Filter", style={'color': '#a16207', 'font-size': '1.5rem', 'margin-top': '0'}),
                            dcc.Dropdown(
                                id='bucket-selector-checklist',
                                options=[{'label': name, 'value': name} for name in bucket_names],
                                value=bucket_names,
                                multi=True,
                                clearable=True,
                                placeholder="Select one or more buckets to filter schemes..."
                            ),
                            html.P(id='filtered-scheme-count', style={'margin-top': '15px', 'font-weight': '600', 'color': '#a16207'})
                        ]
                    ),
                ]
            ),

            # --- Scheme Selection and Total Budget Card ---
            html.Div(
                style={'display': 'grid', 'grid-template-columns': '2fr 1fr', 'gap': '20px', 'margin-bottom': '20px'},
                children=[
                    html.Div(
                        style={'background-color': '#ffffff', 'padding': '15px', 'border-radius': '8px', 'border': '1px solid #e5e7eb', 'box-shadow': '0 4px 6px rgba(0,0,0,0.05)'},
                        children=[
                            html.H3("🎯 Select Scheme for Detail View", style={'color': '#4c1d95', 'font-size': '1.5rem', 'margin-top': '0'}),
                            dcc.Dropdown(
                                id='scheme-selector-dropdown',
                                options=[{'label': name, 'value': name} for name in initial_scheme_options.values()],
                                value=initial_scheme_name,
                                clearable=False,
                                style={'margin-bottom': '10px'}
                            )
                        ]
                    ),
                    # Scorecard 1: Total Allocated to Selected Scheme
                    html.Div(
                        id='scorecard-scheme-budget-container',
                        style={'background-color':'#fff4e5','padding':'15px','border-radius':'8px','text-align':'center', 'border': '1px solid #e5e7eb', 'box-shadow': '0 4px 6px rgba(0,0,0,0.05)'},
                        children=[
                            html.Div(id='scorecard-scheme-title', style={'font-size':'14px','font-weight':'600','margin-bottom':'2px'}, children=[f"Total Budget for {initial_scheme_name}"]),
                            html.Div("Overall Allocation (Cr)", style={'font-size':'12px','color':'#713f12','margin-bottom':'6px'}),
                            html.Div(id='scorecard-scheme-value', style={'font-size':'24px','font-weight':'700', 'color': '#a16207'}, children=[f'₹ {initial_scheme_budget:.2f} Cr'])
                        ]
                    ),
                ]
            ),

            # --- Scorecard: Total Budget Allocated (Overall) ---
            html.Div(
                style={'display': 'flex', 'gap': '20px', 'margin-bottom': '20px', 'justify-content': 'center'},
                children=[
                    html.Div(
                        id='scorecard-total-budget-container',
                        style={'flex': '0 1 50%', 'background-color':'#ecfdf5','padding':'15px','border-radius':'8px','text-align':'center', 'border': '1px solid #e5e7eb', 'box-shadow': '0 4px 6px rgba(0,0,0,0.05)'},
                        children=[
                            html.Div("Total Budget Allocated to All Schemes", style={'font-size':'15px','font-weight':'600','margin-bottom':'2px'}),
                            html.Div("Across All Regions (Cr)", style={'font-size':'12px','color':'#047857','margin-bottom':'6px'}),
                            html.Div(id='scorecard-total-budget-value', style={'font-size':'24px','font-weight':'700', 'color': '#047857'}, children=[f'₹ {initial_total_allocated:.2f} Cr'])
                        ]
                    ),
                ]
            ),

            # --- Map and Chart Row ---
            html.Div(
                style={'display': 'flex', 'gap': '20px', 'margin-bottom': '20px'},
                children=[
                    # Map Chart (Left Column)
                    html.Div(
                        style={'flex': 1, 'background-color': '#ffffff', 'padding': '10px', 'border-radius': '8px', 'border': '1px solid #e5e7eb', 'box-shadow': '0 4px 6px rgba(0,0,0,0.05)'},
                        children=[
                            html.H3("🗺️ Geographic Allocation", style={'color': '#0d9488', 'font-size': '1.5rem', 'margin-top': '0'}),
                            html.Div(
                                dcc.Loading(
                                    id="loading-map",
                                    type="default",
                                    children=dcc.Graph(
                                        id='impact-map-chart',
                                        figure=initial_map_figure,
                                        config={'modeBarButtonsToAdd': ['lasso2d', 'box_select'], 'displaylogo': False}
                                    )
                                )
                            ),
                            html.Div(
                                id='selected-districts-container',
                                style={'margin-top': '10px', 'padding-top': '10px', 'border-top': '1px solid #eee', 'font-size': '0.9rem', 'min-height': '30px'},
                                children=[html.Div(id='selected-districts-output', children=['Use box or lasso tools on the map to choose regions.'])]
                            )
                        ]
                    )
                    ,
                    # Bar Chart (Right Column)
                    html.Div(
                        style={'flex': 1, 'background-color': '#ffffff', 'padding': '10px', 'border-radius': '8px', 'border': '1px solid #e5e7eb', 'box-shadow': '0 4px 6px rgba(0,0,0,0.05)'},
                        children=[
                            html.H3("📈 District Allocation Summary", style={'color': '#0d9488', 'font-size': '1.5rem', 'margin-top': '0'}),
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
                            html.Div(
                                style={'display': 'flex', 'justify-content': 'flex-start', 'margin-top': '10px', 'padding-top': '10px', 'border-top': '1px solid #eee'},
                                children=[
                                    html.Button(
                                        '🗑️ Clear District Selections',
                                        id='clear-map-selection-button',
                                        n_clicks=0,
                                        style={'background-color': '#ef4444', 'color': 'white', 'padding': '6px 12px', 'border': 'none', 'border-radius': '4px', 'cursor': 'pointer', 'font-weight': '600'}
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            # --- End Map and Chart Row ---

            # --- Final Data Table (Selected Scheme) ---
            html.Div(
                style={'background-color': '#ffffff', 'padding': '15px', 'border-radius': '8px', 'border': '1px solid #e5e7eb', 'margin-top': '20px', 'box-shadow': '0 4px 6px rgba(0,0,0,0.05)'},
                children=[
                    html.H3("📄 District-wise Allocation for Selected Scheme", id='scheme-table-title', style={'color': '#4c1d95', 'font-size': '1.5rem', 'margin-top': '0'}),

                    html.Div(id='filtered-budget-summary', style={'font-weight': 'bold', 'margin-bottom': '10px'}),

                    html.Div(
                        style={'display': 'flex', 'gap': '10px', 'margin-bottom': '15px', 'justify-content': 'flex-end'},
                        children=[
                            html.Button('⬇️ Download CSV', id='btn-download-csv', n_clicks=0, style={'background-color': '#22c55e', 'color': 'white', 'padding': '6px 12px', 'border': 'none', 'border-radius': '4px', 'cursor': 'pointer', 'font-weight': '600'}),
                            html.Button('⬇️ Download Excel', id='btn-download-excel', n_clicks=0, style={'background-color': '#1d4ed8', 'color': 'white', 'padding': '6px 12px', 'border': 'none', 'border-radius': '4px', 'cursor': 'pointer', 'font-weight': '600'})
                        ]
                    ),

                    dash_table.DataTable(
                        id='final-scheme-data-table',
                        columns=initial_scheme_table_columns,
                        data=initial_scheme_table_data,
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                        style_cell={'textAlign': 'left', 'padding': '10px'},
                        style_table={'overflowX': 'auto', 'border': '1px solid #e5e7eb', 'border-radius': '8px'}
                    ),
                ]
            ),

            # --- Full Allocation Table (Expandable) ---
            dcc.Dropdown(
                id='dummy-hidden-dropdown',
                options=[{'label': 'Expand Full Allocation Table', 'value': 'expand'}],
                placeholder="Click to expand Full Allocation Table (All Schemes)",
                style={'margin-top': '20px', 'margin-bottom': '10px'}
            ),
            html.Div(
                id='full-allocation-table-container',
                style={'display': 'none', 'background-color': '#ffffff', 'padding': '15px', 'border-radius': '8px', 'border': '1px solid #e5e7eb', 'box-shadow': '0 4px 6px rgba(0,0,0,0.05)'},
                children=[
                    html.H3("📊 Full Budget Allocation Table (All Schemes)", style={'color': '#0d9488', 'font-size': '1.5rem', 'margin-top': '0'}),
                    html.Div(
                        style={'display': 'flex', 'gap': '10px', 'margin-bottom': '15px', 'justify-content': 'flex-end'},
                        children=[
                            html.Button('⬇️ Download Full CSV', id='btn-download-full-csv', n_clicks=0, style={'background-color': '#22c55e', 'color': 'white', 'padding': '6px 12px', 'border': 'none', 'border-radius': '4px', 'cursor': 'pointer', 'font-weight': '600'}),
                            html.Button('⬇️ Download Full Excel', id='btn-download-full-excel', n_clicks=0, style={'background-color': '#1d4ed8', 'color': 'white', 'padding': '6px 12px', 'border': 'none', 'border-radius': '4px', 'cursor': 'pointer', 'font-weight': '600'})
                        ]
                    ),
                    dash_table.DataTable(
                        id='full-allocation-data-table',
                        data=initial_full_alloc_data,
                        # Columns defined implicitly via data until we get a stable column list
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                        style_cell={'textAlign': 'left', 'padding': '10px'},
                        style_table={'overflowX': 'auto', 'border': '1px solid #e5e7eb', 'border-radius': '8px', 'max-height': '400px'}
                    ),
                ]
            ),


            # --- Hidden Divs/Stores for Data Storage and Interactivity ---
            html.Div(id='data-stores', style={'display': 'none'}, children=buckets_data_json),
            dcc.Store(id='selected-districts-store', data=[]),
            # Store for the scheme name to ensure table/title consistency
            dcc.Store(id='selected-scheme-name-store', data=initial_scheme_name),
            # Store for the full, current budget allocation DataFrame (by model)
            dcc.Store(id='full-district-allocations-store', data=initial_district_allocations.to_json(orient='records')),
            # Store for the currently selected scheme's table data (unfiltered)
            dcc.Store(id='full-scheme-table-store', data=initial_scheme_table_df.to_json(orient='records')),
            # Download Components
            dcc.Download(id="download-data-file"),
            dcc.Download(id="download-full-data-file"),
        ]
    )

    # --- CALLBACK: Map Interactivity (Selection Store) ---
    @app.callback(
        [Output('selected-districts-store', 'data'),
         Output('selected-districts-output', 'children')],
        [Input('impact-map-chart', 'selectedData'),
         Input('clear-map-selection-button', 'n_clicks')],
        [State('data-stores', 'children')]
    )
    def update_selected_districts_list(selectedData, n_clicks, buckets_data_json):
        """Manages the list of selected districts and clears selection."""
        ctx = callback_context
        data = json.loads(buckets_data_json)
        geo_unit_col = data.get("geo_unit_col", "District")

        # Clear selection if button clicked
        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'clear-map-selection-button.n_clicks':
            return [], 'Use box or lasso tools on the map to choose regions.'

        selected_districts = []

        if selectedData and 'points' in selectedData:
            for point in selectedData['points']:
                # District name is typically stored in 'hovertext' for choropleth
                district_name = point.get('hovertext')

                if not district_name:
                    # Fallback to text label for districts with no value/hovertext
                    text_label = point.get('text')
                    if text_label and isinstance(text_label, str):
                        # Extract district name from 'District Name (Value)' format
                        district_name = text_label.split('(')[0].strip()

                if district_name and district_name not in selected_districts:
                    selected_districts.append(district_name)

        # Update the output text display
        if selected_districts:
            selected_list_content = html.P(
                [
                    html.Strong(f"Selected {geo_unit_col.replace('_', ' ').title()}s ({len(selected_districts)}): "),
                    html.Span(f"{', '.join(selected_districts)}", title=", ".join(selected_districts), style={'font-weight': 'normal'})
                ],
                style={'margin': 0}
            )
        else:
            selected_list_content = "Use box or lasso tools on the map to choose regions."

        return selected_districts, selected_list_content

    # --- CALLBACK: Full Logic Chain (Model/Bucket/Scheme Change) ---
    @app.callback(
        [Output('scheme-selector-dropdown', 'options'),
         Output('scheme-selector-dropdown', 'value'),
         Output('scorecard-total-budget-value', 'children'),
         Output('full-district-allocations-store', 'data'),
         Output('filtered-scheme-count', 'children')],
        [Input('model-selector-dropdown', 'value'),
         Input('bucket-selector-checklist', 'value')],
        [State('data-stores', 'children')]
    )
    def update_budget_allocations_and_scheme_options(selected_model_name, selected_buckets, buckets_data_json):
        """
        1. Recalculate full budget allocations based on model.
        2. Filter scheme options based on buckets.
        3. Update total budget scorecard.
        """
        data = json.loads(buckets_data_json)
        df_final = pd.read_json(data["original_df_json"], orient='split')
        feature_importances = data["feature_importances"]
        feature_schemes = data["feature_schemes"]
        total_budget = data["total_budget"]
        allocation_mode = data["allocation_mode"]
        geo_col = data["geo_unit_col"]
        feature_buckets_grouped = data["grouped"]

        # A. Get Model Importances
        importances_raw = feature_importances.get(selected_model_name, {})
        if isinstance(importances_raw, pd.DataFrame):
            model_importances = dict(zip(importances_raw["Feature"], importances_raw["Importance"]))
        else:
            model_importances = importances_raw

        # B. Calculate Full Allocation
        feature_budgets, district_allocations = compute_budget_allocation(
            df_final, model_importances, total_budget, geo_col, allocation_mode
        )

        # C. Filter Schemes by Bucket
        available_factors = [f for f, b in feature_budgets.items() if b is not None and b > 0 and f in df_final.columns]

        filtered_feats = []
        if not selected_buckets or len(selected_buckets) == len(feature_buckets_grouped):
            filtered_feats = available_factors
        else:
            for bucket in selected_buckets:
                features = feature_buckets_grouped.get(bucket, [])
                filtered_feats.extend(features)

        filtered_feats = list(set(filtered_feats) & set(available_factors))

        # D. Prepare Scheme Dropdown Options
        scheme_options = {}
        for factor in filtered_feats:
            scheme_options[factor] = feature_schemes.get(factor, factor)

        scheme_dropdown_options = [{'label': name, 'value': name} for name in scheme_options.values()]
        new_scheme_value = list(scheme_options.values())[0] if scheme_options else "No Schemes"

        # E. Update Total Budget Scorecard
        total_allocated = district_allocations["Total_Budget"].sum()
        total_budget_output = f'₹ {total_allocated:.2f} Cr'

        # F. Update Filter Count
        count_output = f"Number of available schemes: {len(scheme_dropdown_options)}"

        return (
            scheme_dropdown_options,
            new_scheme_value,
            total_budget_output,
            district_allocations.to_json(orient='records'),
            count_output
        )

    # --- CALLBACK: Scheme Selection and Chart/Table Update ---
    @app.callback(
        [Output('impact-map-chart', 'figure'),
         Output('impact-bar-chart', 'figure'),
         Output('full-scheme-table-store', 'data'),
         Output('selected-scheme-name-store', 'data'),
         Output('scorecard-scheme-value', 'children'),
         Output('scorecard-scheme-title', 'children'),
         Output('full-allocation-data-table', 'data')], # Update full table data
        [Input('scheme-selector-dropdown', 'value'),
         Input('selected-districts-store', 'data')], # For preserving selection on scheme change
        [State('full-district-allocations-store', 'data'),
         State('data-stores', 'children')]
    )
    def update_scheme_details(selected_scheme_name, selected_districts, district_allocations_json, buckets_data_json):
        """Filters the full allocation data for the selected scheme and generates charts/stores."""
        data = json.loads(buckets_data_json)
        geo_col = data["geo_unit_col"]
        GEO_COL = data["GEO_COL"]
        feature_schemes = data["feature_schemes"]
        df_final = pd.read_json(data["original_df_json"], orient='split')

        empty_map = go.Figure(layout=dict(title=None, height=500))
        empty_bar = px.bar(title=None, height=500)
        empty_df_json = pd.DataFrame().to_json(orient='records')

        # 1. Load Data
        if not district_allocations_json:
            return empty_map, empty_bar, empty_df_json, selected_scheme_name, 'N/A', f"Total Budget for {selected_scheme_name}", []

        district_allocations = pd.read_json(district_allocations_json, orient='records')

        # 2. Find Factor Name (Reverse Lookup)
        reverse_feature_schemes = {v: k for k, v in feature_schemes.items()}
        selected_factor_normalized = reverse_feature_schemes.get(selected_scheme_name)
        if not selected_factor_normalized or selected_factor_normalized not in district_allocations.columns:
            # Fallback if scheme name is the factor name
            selected_factor_normalized = selected_scheme_name
            if selected_factor_normalized not in district_allocations.columns:
                return empty_map, empty_bar, empty_df_json, selected_scheme_name, 'N/A', f"Total Budget for {selected_scheme_name}", []

        # 3. Scheme Budget Scorecard
        initial_scheme_budget = district_allocations[selected_factor_normalized].sum()
        scheme_budget_output = f'₹ {initial_scheme_budget:.2f} Cr'
        scheme_title_output = f"Total Budget for {selected_scheme_name}"

        # 4. Scheme-Specific Data Preparation (Full data for charts)
        change_df = district_allocations[[geo_col, selected_factor_normalized]].copy()
        change_df.rename(columns={selected_factor_normalized: "Change"}, inplace=True)

        # 5. Generate Charts (Map uses selection store data to maintain visual selection)
        map_figure = generate_map(change_df, gdf, geo_col, GEO_COL, "Geographic Distribution of Scheme Budget", selected_districts)
        bar_figure = generate_bar_chart(change_df, geo_col, f"{selected_scheme_name} Allocation by {geo_col.replace('_', ' ').title()}")

        # 6. Scheme Table Data Store (Unfiltered)
        scheme_table_df = change_df[[geo_col, "Change"]].copy()
        scheme_table_df.columns = [geo_col, f"{selected_scheme_name} Budget"]
        scheme_table_data_json = scheme_table_df.to_json(orient='records')

        # 7. Update Full Allocation Table (Rename columns for display)
        full_alloc_df = district_allocations.copy()
        rename_map = {col: feature_schemes.get(col, col) for col in full_alloc_df.columns if col not in [geo_col, "Total_Budget"]}
        full_alloc_df.rename(columns=rename_map, inplace=True)
        full_alloc_data = full_alloc_df.to_dict('records')


        return (
            map_figure,
            bar_figure,
            scheme_table_data_json,
            selected_scheme_name,
            scheme_budget_output,
            scheme_title_output,
            full_alloc_data
        )

    # --- CALLBACK: Dynamic Table and Bar Chart Filtering (Based on Map Selection) ---
    @app.callback(
        [Output('final-scheme-data-table', 'data'),
         Output('final-scheme-data-table', 'columns'),
         Output('filtered-budget-summary', 'children')],
        [Input('selected-districts-store', 'data')],
        [State('full-scheme-table-store', 'data'),
         State('selected-scheme-name-store', 'data'),
         State('data-stores', 'children')]
    )
    def filter_table_data(selected_districts, scheme_table_json, scheme_name, buckets_data_json):
        """Filters the selected scheme's table data based on the map selection."""
        if not scheme_table_json:
            return [], [], 'Total Budget allocated to selected districts: ₹ 0.00 Cr'

        data = json.loads(buckets_data_json)
        geo_col = data["geo_unit_col"]

        full_df = pd.read_json(scheme_table_json, orient='records')
        budget_col_name = f"{scheme_name} Budget"

        # Define columns explicitly for the output table
        columns = [
            {"name": geo_col, "id": geo_col},
            {"name": budget_col_name, "id": budget_col_name, 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)}
        ]

        if not selected_districts:
            filtered_df = full_df
            summary_prefix = "Total Budget allocated to all regions:"
        else:
            filtered_df = full_df[full_df[geo_col].isin(selected_districts)].copy()
            summary_prefix = f"Total Budget allocated to selected districts ({len(selected_districts)}):"

        filtered_total_budget = filtered_df[budget_col_name].sum()
        summary_output = html.Div(
            f"{summary_prefix} ₹ {filtered_total_budget:.2f} Cr",
            style={'color': '#a16207', 'font-size': '1rem'}
        )

        return filtered_df.to_dict('records'), columns, summary_output


    # --- CALLBACK: Download Handlers (Selected Scheme) ---
    @app.callback(
        Output("download-data-file", "data"),
        [Input("btn-download-csv", "n_clicks"),
         Input("btn-download-excel", "n_clicks")],
        [State('final-scheme-data-table', 'data'),
         State('selected-scheme-name-store', 'data')]
    )
    def download_scheme_data(n_clicks_csv, n_clicks_excel, table_data, scheme_name):
        """Triggers the download of the currently filtered scheme data table."""
        ctx = callback_context
        if not ctx.triggered or not table_data:
            return no_update

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        df_to_export = pd.DataFrame(table_data)

        if button_id == 'btn-download-csv':
            return dcc.send_data_frame(
                df_to_export.to_csv,
                filename=f"{scheme_name.replace(' ', '_')}_budget.csv",
                index=False,
            )
        elif button_id == 'btn-download-excel':
            return dcc.send_data_frame(
                df_to_export.to_excel,
                filename=f"{scheme_name.replace(' ', '_')}_budget.xlsx",
                index=False,
                sheet_name="Selected Scheme Budget"
            )

        return no_update

    # --- CALLBACK: Download Handlers (Full Allocation) ---
    @app.callback(
        Output("download-full-data-file", "data"),
        [Input("btn-download-full-csv", "n_clicks"),
         Input("btn-download-full-excel", "n_clicks")],
        [State('full-allocation-data-table', 'data')]
    )
    def download_full_data(n_clicks_csv, n_clicks_excel, table_data):
        """Triggers the download of the full allocation data table."""
        ctx = callback_context
        if not ctx.triggered or not table_data:
            return no_update

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        df_to_export = pd.DataFrame(table_data)

        if button_id == 'btn-download-full-csv':
            return dcc.send_data_frame(
                df_to_export.to_csv,
                filename="full_district_allocations.csv",
                index=False,
            )
        elif button_id == 'btn-download-full-excel':
            return dcc.send_data_frame(
                df_to_export.to_excel,
                filename="full_district_allocations.xlsx",
                index=False,
                sheet_name="All Scheme Budgets"
            )

        return no_update

    # --- CALLBACK: Toggle Full Allocation Table ---
    @app.callback(
        Output('full-allocation-table-container', 'style'),
        [Input('dummy-hidden-dropdown', 'value')],
        [State('full-allocation-table-container', 'style')]
    )
    def toggle_full_table(dropdown_value, current_style):
        """Toggles the visibility of the full allocation table."""
        if dropdown_value == 'expand':
            current_style['display'] = 'block'
            return current_style
        return {'display': 'none'}


    return app

# ==============================================================================
# 5. SERVER THREAD TARGET FUNCTION
# ==============================================================================

def run_dash_server_thread_target(df_json: str, app_state_json: str, geo_col: str, port: int):
    """The entry point for the Dash server thread."""
    try:
        # 1. Deserialize JSON data passed from Streamlit thread
        df_final = pd.read_json(df_json, orient='split')
        app_state = json.loads(app_state_json)

        # 2. Retrieve GeoDataFrame separately
        gdf = None
        gdf_ref = app_state.get('gdf_ref')
        if gdf_ref and gdf_ref in PYTHON_OBJECT_LOOKUP:
            try:
                gdf = PYTHON_OBJECT_LOOKUP[gdf_ref]
            except Exception as e:
                print(f"Error retrieving GDF during initialization: {e}")

        # 3. Create the Dash application
        dash_app = create_dash_app(df_final, app_state, geo_col, gdf)

        # 4. Run the server
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
