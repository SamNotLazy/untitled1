import json

import Dash_Pages.dash_prescriptive_modelling as dpm
from dash import Dash, html, dcc, callback_context, no_update
from dash.dependencies import Input, Output, State, ALL
from typing import Dict, Any, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Assuming dpm and PYTHON_OBJECT_LOOKUP exist in the environment
# class dpm:
#     PYTHON_OBJECT_LOOKUP = {}

def generate_mini_map(change_df: pd.DataFrame, app_state: dict| None, selected_districts: list | None = None) -> px.choropleth_mapbox:
    """
    Generates the Plotly Mapbox Choropleth map.
    """
    if app_state is None:
        app_state = {}

    gdf = None
    gdf_ref = app_state.get('gdf_ref')
    if gdf_ref and gdf_ref in dpm.PYTHON_OBJECT_LOOKUP:
        try:
            # Note: dpm.PYTHON_OBJECT_LOOKUP is assumed to be defined externally
            gdf = dpm.PYTHON_OBJECT_LOOKUP[gdf_ref]
        except Exception as e:
            print(f"Error retrieving GDF from lookup: {e}")
    # print(gdf)
    # Ensure selected_districts is an iterable list, even if None is passed
    selected_districts = selected_districts or []

    plot_df = change_df.copy()
    geo_key = app_state.get("geo_unit_col", "N/A")
    map_col = "Change"
    direction = app_state.get("target_direction", "Increase")
    GEO_COL = app_state.get("GEO_COL", "N/A")

    # -----------------------------------------------------------------
    # Correctly filter GDF using .isin()
    # and only if selected_districts is not empty.
    # -----------------------------------------------------------------
    if gdf is not None and not gdf.empty and selected_districts and GEO_COL in gdf.columns:
        # Filter the GeoDataFrame to include only the selected districts
        gdf = gdf[gdf[GEO_COL]==selected_districts].copy()


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

    fixed_height = 180

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

    # --- Geographic Bounds and Centering (Calculated on the filtered GDF) ---
    minx, miny, maxx, maxy = plot_df.total_bounds
    center_lon = (minx + maxx) / 2
    center_lat = (miny + maxy) / 2

    width = abs(maxx - minx)
    height = abs(maxy - miny)
    max_dim = max(width, height)

    initial_zoom = float(gdf['_optimal_zoom'])


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


    map_fig.update_mapboxes(
        bearing=0,
        pitch=0,
        # Ensure the final mapbox settings use the calculated center and zoom
        center={"lat": center_lat, "lon": center_lon},
        accesstoken=None, # Clear access token
        layers=[],
        zoom=initial_zoom
    )

    map_fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title=None,
        height=fixed_height,
        # Hide the continuous color scale legend
        coloraxis_showscale=False,
        # Hide any trace legends (like the 'Labels' trace)
        showlegend=False
    )




    return map_fig

def generate_detail_page_layout(selected_districts: List[str]) -> html.Div:
    """
    Generates the layout for the detailed prescriptions page (/details route).

    It includes the main context map (loading from store), a summary of feature
    adjustments (loading from store), and the individual district detail panels.
    """
    print(f"DEBUG: [LAYOUT] Generating detail page layout. Initial selected districts: {selected_districts}")
    district_list = ", ".join(selected_districts) if selected_districts else "No districts selected."

    return html.Div(
        style={
            'max-width': '1400px',
            'margin': '20px auto',
            'padding': '20px',
            'background-color': '#ffffff',
            'border-radius': '10px',
            'box-shadow': '0 0 20px rgba(0, 0, 0, 0.1)'
        },
        children=[
            html.Div(
                style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'border-bottom': '2px solid #93c5fd', 'padding-bottom': '10px', 'margin-bottom': '20px'},
                children=[
                    html.H2("Detailed District Prescriptions Dashboard", style={'color': '#1d4ed8', 'margin': 0}),
                    html.A(
                        html.Button(
                            '⬅️ Go Back to Main Dashboard',
                            id='go-back-button',
                            n_clicks=0,
                            style={
                                'background-color': '#059669',
                                'color': 'white',
                                'padding': '8px 16px',
                                'border': 'none',
                                'border-radius': '4px',
                                'cursor': 'pointer',
                                'font-weight': '600'
                            }
                        ),
                        href='/',
                        style={'text-decoration': 'none'}
                    )
                ]
            ),
            html.P("This page provides a detailed view of the calculated impact for the selected regions, retaining the selection from the main map, which can be modified here.", style={'font-weight': 'normal', 'word-break': 'break-all', 'margin-bottom': '20px'}),

            # NEW SECTION: Map (Context) and Stacked Charts (Detail)
            html.H3("Regional Impact Analysis", style={'color': '#1e3a8a', 'margin-top': '20px', 'padding-top': '10px', 'border-top': '1px solid #e5e7eb'}),
            html.Div(
                style={'display': 'flex', 'gap': '10px', 'margin-bottom': '20px'},
                children=[
                    # Map Chart (Left, full height for context) & FEATURE IMPACT PANEL (BELOW MAP)
                    html.Div(
                        style={'flex': 1, 'background-color': '#ffffff', 'padding': '0', 'border-radius': '8px'},
                        children=[
                            # 1. Map Chart (Context from main dashboard)
                            html.Div(
                                style={'background-color': '#f9f9f9', 'padding': '15px', 'border-radius': '8px', 'border': '1px solid #e5e7eb', 'margin-bottom': '20px'},
                                children=[
                                    html.H4("Geographic Impact ", style={'color': '#1d4ed8', 'font-size': '1.25rem', 'margin-top': '0'}),
                                    dcc.Loading(
                                        id="loading-detail-map",
                                        type="default",
                                        children=dcc.Graph(
                                            id='detail-impact-map-chart',
                                            config={
                                                'displaylogo': False,
                                                'scrollZoom': True,
                                                'modeBarButtonsToAdd': ['lasso2d', 'box_select'] # Add tools for selection on detail page
                                            },
                                            style={'height': '280px'}
                                        )
                                    )
                                ]
                            ),

                            # 2. FEATURE IMPACT PANEL (PLACED BELOW THE MAP, INSIDE THE LEFT COLUMN DIV)
                            html.Div(
                                style={'background-color': '#f9f9f9', 'padding': '15px', 'border-radius': '8px', 'border': '1px solid #e5e7eb', 'max-height': '300px', 'overflowY': 'auto'},
                                children=[
                                    html.H4("Prescriptive Feature Adjustments (Averages)", style={'color': '#1e3a8a', 'font-size': '1.25rem', 'margin-top': '0'}),

                                    dcc.Loading(
                                        id="loading-detail-duel-bar",
                                        type="default",
                                        children=html.Div(
                                            id='detail-feature-impact-panel', # New ID for the Feature Impact content
                                            style={'height': '50%'}
                                        )
                                    )
                                ]
                            )
                        ]
                    ),

                    # Individual District List (Right, Stacked)
                    html.Div(
                        style={'flex': 1, 'background-color': '#ffffff', 'padding': '0', 'border-radius': '8px'},
                        children=[
                            html.Div(
                                style={'background-color': '#f9f9f9', 'padding': '15px', 'border-radius': '8px', 'border': '1px solid #e5e7eb', 'min-height': '720px'},
                                children=[
                                    html.H4("Selected Districts: Prescribed Adjustments", style={'color': '#059669', 'font-size': '1.25rem', 'margin-top': '0'}),
                                    # CUSTOM LEGEND (Copied from main panel)
                                    html.Div(
                                        style={'font-size': '0.75rem', 'display': 'flex', 'gap': '5px', 'margin-bottom': '5px', 'justify-content': 'center'},
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
                                    # --- PRIMARY MULTI-SELECT FILTER (RENAMED) ---
                                    html.Div(
                                        [
                                            html.P("Districts to Display/Filter:", style={'font-size': '0.9rem', 'font-weight': '600', 'margin-bottom': '5px', 'color': '#1d4ed8'}),
                                            dcc.Dropdown(
                                                id='detail-district-filter', # RENAMED ID
                                                multi=True,
                                                placeholder="Select districts to filter the detail view..."
                                            ),
                                            html.Div(
                                                id='single-district-output',
                                                style={'display': 'none'} # Hidden/unused
                                            ),
                                            html.Hr(style={'margin': '10px 0', 'border-top': '1px solid #bfdbfe'})
                                        ], style={'margin-bottom': '10px'}
                                    ),
                                    # --- END PRIMARY MULTI-SELECT FILTER ---

                                    dcc.Loading(
                                        id="loading-individual-maps",
                                        type="default",
                                        children=html.Div(
                                            id='individual-district-maps-container',
                                            style={'max-height': '500px', 'overflowY': 'scroll', 'padding-right': '10px'}
                                        )
                                    )
                                ]
                            ),
                        ]
                    ),
                ]
            ),
            # New, dedicated local store for selection on this page only
            dcc.Store(id='detail-local-selection-store', data=selected_districts)
        ]
    )

def register_detail_page_callbacks(app: Dash,app_state):
    """
    Registers all callbacks specific to the /details page.
    """

    # --- CALLBACK 1: Initialize Dropdown Options and Value (One-Time Load) ---
    @app.callback(
        [Output('detail-district-filter', 'options'), # Output for the consolidated multi-select options
         Output('detail-district-filter', 'value'),   # Output for the consolidated multi-select value
         Output('detail-local-selection-store', 'data', allow_duplicate=True)], # Initialize local store
        [Input('url', 'pathname')],
        [State('full-impact-data-store', 'data'),
         State('selected-districts-store', 'data')],
        prevent_initial_call='initial_duplicate'
    )
    def initialize_detail_district_dropdown(pathname, impact_data_json, selected_districts_from_store):
        """
        Populates the primary multi-select options and sets the initial value
        for the dropdown and the new local store based on the global store.
        """
        print(f"DEBUG: [CB1] Initializing dropdown and local store. Pathname: {pathname}")
        if pathname == '/details':
            print(f"DEBUG: [CB1] Path is /details. impact_data_json len: {len(impact_data_json) if impact_data_json else 0}")
            try:
                impact_df = pd.read_json(impact_data_json, orient='records')
                # Assumes 'GeoUnitName' column exists in the stored impact data
                all_districts = impact_df['GeoUnitName'].unique().tolist()
                print(f"DEBUG: [CB1] Successfully loaded {len(all_districts)} districts.")

                options = [{'label': d, 'value': d} for d in all_districts]

                # --- MODIFICATION TO DE-HIGHLIGHT ON LOAD ---
                # The local store is initialized to an empty list, ensuring no highlighting on initial load.
                initial_value = []
                print(f"DEBUG: [CB1] Initial value FORCED to empty list to de-highlight regions on load.")
                # ---------------------------------------------

                # Return options for the primary multi-select, its initial value, and the initial local store data
                return options, initial_value, initial_value
            except Exception as e:
                print(f"ERROR: [CB1] Failed to initialize dropdown: {e}")
                # Fallback on error
                return [], [], []

        print("DEBUG: [CB1] Pathname not /details. Returning no_update.")
        # Note: Reduced the number of returned values to 3 to match the new outputs
        return no_update, no_update, no_update


    # --- CALLBACK 2 (MASTER): Unified Selection Handler ---
    # This callback updates the single source of truth ('detail-local-selection-store')
    # and synchronizes the global store and UI elements.
    @app.callback(
        [Output('detail-local-selection-store', 'data', allow_duplicate=True), # Local store (New primary source)
         Output('selected-districts-store', 'data', allow_duplicate=True),      # Global store (Sync for persistence)
         Output('detail-district-filter', 'value', allow_duplicate=True), # Dropdown UI
         Output('detail-selected-districts-output', 'children', allow_duplicate=True)], # Text output
        [Input('detail-district-filter', 'value'), # Now the only dropdown input
         Input('detail-impact-map-chart', 'selectedData'),
         Input('clear-detail-map-selection-button', 'n_clicks')],
        [State('full-impact-data-store', 'data'),
         State('app-state-data', 'children')],
        prevent_initial_call=True
    )
    def update_selection_master(dropdown_value: List[str], selectedData: Dict[str, Any], n_clicks: int, impact_data_json: str, app_state_json: str):
        """
        Master callback to handle all selection inputs (Dropdown, Map, Clear Button)
        and determine the single source of truth for selected districts.
        """
        ctx = callback_context
        # Use a more robust check for initial/no trigger
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered and ctx.triggered[0]['prop_id'] else "No Trigger"

        # --- Load necessary state data ---
        geo_unit_col_name = "N/A"
        impact_df = pd.DataFrame()
        try:
            app_state = json.loads(app_state_json)
            geo_unit_col_name = app_state.get("geo_unit_col", "N/A")
            if impact_data_json:
                impact_df = pd.read_json(impact_data_json, orient='records')
        except Exception as e:
            print(f"ERROR: [CB_MASTER] Failed to load state data: {e}")
            # If state data fails to load, we can't reliably process map interaction, return no_update for safety
            return no_update, no_update, no_update, no_update


        new_selected_districts = []

        # --- Logging triggered action ---
        print(f"DEBUG: [CB_MASTER] Triggered ID: {triggered_id}. n_clicks: {n_clicks}. selectedData present: {bool(selectedData)}. Dropdown value count: {len(dropdown_value) if dropdown_value else 0}")

        if triggered_id == 'clear-detail-map-selection-button':
            # Case 1: Clear button clicked (Highest priority action)
            print("DEBUG: [CB_MASTER] Clear button action: Clearing all selections.")
            new_selected_districts = []

        elif triggered_id == 'detail-district-filter':
            # Case 2: Dropdown changed (Authoritative selection list)
            # The dropdown now controls both the store state and the map highlights
            print(f"DEBUG: [CB_MASTER] Dropdown action: {len(dropdown_value) if dropdown_value else 0} selected.")
            if dropdown_value is not None:
                new_selected_districts = list(dropdown_value)

        elif triggered_id == 'detail-impact-map-chart':
            # Case 3: Map selection tool used (Box/Lasso select)
            print(f"DEBUG: [CB_MASTER] Map action. Selected data structure received.")

            # 3.1 Handle clearing via map interaction (clicking outside selection or using map tool to select nothing)
            if selectedData is None or (isinstance(selectedData, dict) and not selectedData.get('points')):
                print("DEBUG: [CB_MASTER] Map selection cleared via map interaction (selectedData is None or empty points list).")
                new_selected_districts = []

            # 3.2 Handle new selection via map interaction
            elif isinstance(selectedData, dict) and 'points' in selectedData and selectedData['points'] and not impact_df.empty:

                selected_indices = [point.get('pointIndex') for point in selectedData.get('points', []) if point.get('pointIndex') is not None]

                if 'GeoUnitName' in impact_df.columns:
                    # Filter the original DataFrame by the collected indices
                    valid_indices = [idx for idx in selected_indices if idx < len(impact_df)]
                    if valid_indices:
                        selected_rows = impact_df.iloc[valid_indices]
                        new_selected_districts = selected_rows['GeoUnitName'].unique().tolist()
                        print(f"DEBUG: [CB_MASTER] Map extraction: {len(new_selected_districts)} districts extracted.")
                else:
                    print("ERROR: [CB_MASTER] Cannot process map selection: 'GeoUnitName' column missing.")

        else:
            # Fallback - Should not be hit now.
            print(f"DEBUG: [CB_MASTER] Unhandled trigger: {triggered_id}. Returning no_update.")
            return no_update, no_update, no_update, no_update

        # --- Update Display Output ---
        if not new_selected_districts:
            output_content = "Use map or dropdown to select regions."
        else:
            output_content = html.P(
                [
                    html.Strong(f"Selected {geo_unit_col_name.replace('_', ' ').title()}s ({len(new_selected_districts)}): "),
                    html.Span(f"{', '.join(new_selected_districts)}", title=", ".join(new_selected_districts), style={'font-weight': 'normal'})
                ],
                style={'margin': 0}
            )

        # Ensure the store and dropdown value is always updated based on the winning list
        store_update = list(new_selected_districts) # Use list() to ensure Dash sees a new object reference

        print(f"DEBUG: [CB_MASTER] FINAL STATE: {len(store_update)} districts selected. Syncing stores and output.")
        # Return: Local Store, Global Store, Dropdown Value, Text Output
        return store_update, store_update, store_update, output_content


    # --- CALLBACK 3: Map Renderer (Replaces sync_store_to_map_selection and original display_detail_map) ---
    @app.callback(
        Output('detail-impact-map-chart', 'figure'),
        [Input('detail-local-selection-store', 'data')], # Triggered by any selection change (CB2)
        [State('full-impact-data-store', 'data'),
         State('map-figure-store', 'data')],
        prevent_initial_call='initial_duplicate'
    )
    def update_detail_map_figure(selected_districts: List[str], impact_data_json: str, map_figure_json: str):
        """
        Loads the base map figure and re-renders it with the current selection state (highlights)
        applied to the Plotly trace's selectedpoints property.
        """
        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'] if ctx.triggered else "Not Triggered"
        print(f"DEBUG: [CB3 - Map Renderer] Trigger: {triggered_id}. Local Store districts: {len(selected_districts) if selected_districts else 0}")
        if not map_figure_json:
            print("DEBUG: [CB3] No base map figure in store. Returning no_update.")
            return no_update

        try:
            # 1. Load the existing figure and data
            figure = go.Figure(json.loads(map_figure_json))
            # Access the layout's mapbox center dictionary
            center_dict = figure.layout.mapbox.center

            # Extract the latitude and longitude
            current_lat = center_dict['lat']
            current_lon = center_dict['lon']
            figure.update_mapboxes(
                zoom=4.25,
                center={"lat": current_lat+3, "lon": current_lon}
            )
            # Resize the figure
            figure.update_layout(
                height=270,
            )

            impact_df = pd.read_json(impact_data_json, orient='records')

        except Exception as e:
            print(f"ERROR: [CB3] Failed to load figure or impact data: {e}")
            return no_update

        # --- Calculate selected indices ---
        if not selected_districts:
            selected_indices = None # Clear selection
            print("DEBUG: [CB3] Selected districts store is empty. Clearing map selection.")
        else:
            # Map selected district names back to the DataFrame index used to generate the map
            selected_indices = impact_df[impact_df['GeoUnitName'].isin(selected_districts)].index.tolist()
            print(f"DEBUG: [CB3] Matched {len(selected_indices) if selected_indices else 0} indices to selected districts.")
            if not selected_indices:
                selected_indices = None
            else:
                # Ensure indices are integers for Plotly's selectedpoints property
                selected_indices = [int(i) for i in selected_indices]


        # --- Apply selected indices to the trace ---
        # Assuming the main map data is the first trace (index 0)
        if figure.data and figure.data[0]:
            trace_index = 0

            # MODIFICATION: Explicitly remove any residual selectedpoints property from the imported
            # figure data if the calculated selection is None. This guarantees de-highlighting,
            # directly addressing the user's concern about the imported "mapjson" having highlights.
            if selected_indices is None and hasattr(figure.data[trace_index], 'selectedpoints'):
                # Plotly's internal structure can retain the key even if the value is None.
                # Deleting the property ensures a clean render state for the imported figure.
                figure.data[trace_index]['selectedpoints']=[]
                print(f"DEBUG: [CB3] Removed residual 'selectedpoints' from imported figure trace.")


            # Set the selectedpoints property directly on the trace (to None or a list of indices)
            figure.data[trace_index].selectedpoints = selected_indices

            # Use uirevision to prevent unnecessary layout resets but still allow trace updates
            # figure.update_layout(uirevision=)

        print(f"DEBUG: [CB3] Re-rendering map with new selection state.")

        return figure



    # --- CALLBACK 6: Feature Impact Panel for Detail Page (Original CB8) ---
    @app.callback(
        Output('detail-feature-impact-panel', 'children'),
        [Input('url', 'pathname')],
        [State('intervened-feature-list-store', 'data'),
         State('feature-averages-store', 'data')]
    )
    def update_detail_feature_impact_panel(pathname, selected_feats: list, feature_averages_json: str):
        """
        Generates the dual bar charts for Original vs. Modified feature averages on the detail page.
        This uses the data calculated and stored by the main dashboard.
        """
        print(f"DEBUG: [CB6] Feature Impact Panel triggered. Pathname: {pathname}")
        if pathname != '/details':
            print("DEBUG: [CB6] Not on /details. Returning no_update.")
            return no_update

        if not selected_feats:
            print("DEBUG: [CB6] No features selected for intervention. Returning prompt.")
            return html.P("No features selected for intervention.", style={'color': '#4a5568', 'margin': '10px'})

        try:
            print(f"DEBUG: [CB6] Loading feature averages JSON. Len: {len(feature_averages_json) if feature_averages_json else 0}")
            feature_averages = json.loads(feature_averages_json) if feature_averages_json else {}
        except json.JSONDecodeError as e:
            print(f"ERROR: [CB6] Error loading feature average data: {e}")
            return html.P("Error loading feature average data.", style={'color': '#ef4444', 'margin': '10px'})

        chart_elements = []

        for feat in selected_feats:
            avg_data = feature_averages.get(feat, {'original': 0.0, 'modified': 0.0})
            old_val = avg_data.get('original', 0.0)
            new_val = avg_data.get('modified', 0.0)
            print(f"DEBUG: [CB6] Processing feature '{feat}': Old Avg={old_val:.4f}, New Avg={new_val:.4f}")

            plain_text_label = feat.replace('_', ' ').title()

            max_abs_val = max(abs(old_val), abs(new_val))
            x_range_max = max(1.0, max_abs_val * 1.1)
            min_val = min(old_val, new_val)
            x_range_min = min(-1.0, min_val * 1.1)
            x_range = [x_range_min, x_range_max]

            bar_fig = go.Figure(data=[
                go.Bar(
                    name='Original Avg', x=[old_val], y=[""], orientation='h', marker_color='#1d4ed8',
                    text=[f"{old_val:.4f}"], textposition='inside', insidetextanchor='middle', showlegend=False
                ),
                go.Bar(
                    name='Modified Avg', x=[new_val], y=[""], orientation='h', marker_color='#059669',
                    text=[f"{new_val:.4f}"], textposition='inside', insidetextanchor='middle', showlegend=False
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

            # Note: This chart's ID should match the main page's chart ID structure
            # so the sensitivity modal can be opened from here as well.
            chart_elements.append(
                html.Div(
                    style={'margin-bottom': '10px', 'padding': '0', 'border-bottom': '1px dotted #e5e7eb', 'cursor': 'pointer'},
                    children=[
                        html.P(plain_text_label, style={'font-size': '0.9rem', 'font-weight': '600', 'margin-bottom': '2px', 'margin-top': '2px', 'color': '#2d3748', 'text-align': 'center'}),
                        dcc.Graph(
                            id={'type': 'feature-impact-chart', 'index': feat},
                            figure=bar_fig,
                            config={'displayModeBar': False},
                            style={'height': '70px', 'width': '100%'}
                        )
                    ]
                )
            )

        print(f"DEBUG: [CB6] Finished processing. Returning {len(chart_elements)} feature impact charts.")
        return chart_elements



    # --- CALLBACK 5: Display Selected District List for Detail Page (Original CB7) ---
    @app.callback(
        Output('individual-district-maps-container', 'children'),
        [Input('url', 'pathname'),
         Input('detail-district-filter', 'value')], # NEW INPUT: Primary filter (multi-select)
        [State('full-display-data-store', 'data'),
         State('intervened-feature-list-store', 'data')]
    )
    def display_selected_district_list(pathname, selected_districts_from_filter: List[str], display_data_json: str, intervene_feats: List[str]):
        """
        Generates a list of selected district names, each in its own styled div element.
        Defaults to showing ALL districts if the multi-select filter is empty.
        """
        print(f"DEBUG: [CB5] District list generation triggered. Pathname: {pathname}. Filtered districts count: {len(selected_districts_from_filter) if selected_districts_from_filter else 0}")

        if pathname != '/details':
            print("DEBUG: [CB5] Not on /details. Returning no_update.")
            return no_update

        full_df = pd.DataFrame()
        if display_data_json and display_data_json != '[]':
            try:
                print(f"DEBUG: [CB5] Loading full display data. Len: {len(display_data_json)}")
                full_df = pd.read_json(display_data_json, orient='records')
            except ValueError as e:
                print(f"ERROR: [CB5] Error loading detailed feature data: {e}")
                return html.P("Error loading detailed feature data.", style={'color': '#ef4444', 'padding': '10px'})

        if full_df.empty:
            print("DEBUG: [CB5] Full display DataFrame is empty.")
            return html.P("No detailed feature data available.", style={'color': '#713f12', 'padding': '10px'})

        # NOTE: Using a specific column name if available, otherwise defaulting to the first column.
        geo_unit_col_name = 'GeoUnitName' if 'GeoUnitName' in full_df.columns else full_df.columns[0]
        print(f"DEBUG: [CB5] Geo Unit Column inferred as: {geo_unit_col_name}")

        # --- Logic to prioritize Filter / Default to All ---
        if selected_districts_from_filter:
            # Filter override: Use the list selected in the dropdown
            selected_districts_to_display = selected_districts_from_filter
            print(f"DEBUG: [CB5] Filter activated. Displaying {len(selected_districts_to_display)} districts.")
        else:
            # Default: Show All Districts
            selected_districts_to_display = full_df[geo_unit_col_name].head(5).unique().tolist()
            # selected_districts_to_display=selected_districts_to_display[0]
            print(f"DEBUG: [CB5] No selection filter. Defaulting to show ALL {len(selected_districts_to_display)} districts.")

        # FIX: Filter out any None values before sorting to prevent TypeError.
        selected_districts_to_display = [d for d in selected_districts_to_display if d is not None]

        # Check for empty list after prioritization (should only happen if full_df was non-empty but lacked GeoUnitName data)
        if not selected_districts_to_display:
            print("DEBUG: [CB5] No districts available. Returning prompt.")
            return html.P("No districts available. Please ensure your data is loaded correctly.", style={'color': '#713f12', 'padding': '10px'})

        list_elements = []
        summary_cols = ["Target (Original)", "Predicted", "Change"]
        for col in summary_cols:
            if col in full_df.columns:
                # Ensure columns are numeric for calculation/display
                full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
                print(f"DEBUG: [CB5] Converted column '{col}' to numeric.")

        print(f"DEBUG: [CB5] Intervened features: {intervene_feats}")
        # Sort the districts alphabetically for a clean list view
        for district in sorted(selected_districts_to_display):
            print(f"DEBUG: [CB5] Processing district: {district}")
            district_data_row = full_df[full_df[geo_unit_col_name] == district]

            if district_data_row.empty:
                print(f"DEBUG: [CB5] Data missing for district: {district}")
                list_elements.append(html.P(f"Data missing for district: {district}", className='text-xs italic text-gray-500 p-2'))
                continue

            data_row = district_data_row.iloc[0]

            # --- Target Summary Section ---
            original_target = data_row.get("Target (Original)")
            predicted_target = data_row.get("Predicted")
            change_value = data_row.get("Change")
            print(f"DEBUG: [CB5] Target Summary: Original={original_target}, Predicted={predicted_target}, Change={change_value}")

            summary_children = []
            if pd.isna(change_value):
                change_text = "N/A"
                change_style = {'font-weight': '400', 'color': '#ef4444'}
            else:
                change_text = f"{change_value:+.4f}"
                change_style = {'font-weight': '600', 'color': '#059669' if change_value >= 0 else '#ef4444'}

            summary_children.extend([
                html.Div([html.Span("Target (Original): ", className='font-normal'), html.Span(f"{original_target:.4f}" if not pd.isna(original_target) else "N/A", className='font-bold')], className='text-xs flex justify-between'),
                html.Div([html.Span("Target (Modified): ", className='font-normal'), html.Span(f"{predicted_target:.4f}" if not pd.isna(predicted_target) else "N/A", className='font-bold')], className='text-xs flex justify-between'),
                html.Div([html.Span("Change: ", className='font-normal'), html.Span(change_text, style=change_style)], className='text-sm flex justify-between pt-1 border-t border-blue-200 mt-1')
            ])

            # --- Feature Detail Section (Mini dual bar charts) ---
            feature_details = []
            if intervene_feats:
                for feat in intervene_feats:
                    original_col = f"{feat} (Original)"
                    modified_col = f"{feat} (Modified)"
                    feat_label = feat.replace('_', ' ').title()

                    if original_col in data_row and modified_col in data_row:
                        orig_val = data_row[original_col]
                        mod_val = data_row[modified_col]

                        print(f"DEBUG: [CB5] Feature {feat}: Original={orig_val:.4f}, Modified={mod_val:.4f}")

                        # Check if the feature value was actually modified
                        is_adjusted = not np.isclose(orig_val, mod_val, equal_nan=True)

                        # Determine X-axis range dynamically for the dual bar chart
                        max_abs_val = max(abs(orig_val), abs(mod_val)) if not pd.isna(orig_val) and not pd.isna(mod_val) else 0.1
                        x_range_max = max(0.1, max_abs_val * 1.2)
                        min_val = min(orig_val, mod_val) if not pd.isna(orig_val) and not pd.isna(mod_val) else -0.1
                        x_range_min = min(-0.1, min_val * 1.1)

                        # --- Mini Dual Bar Chart Figure ---
                        bar_fig = go.Figure(data=[
                            go.Bar(
                                name='Original', x=[orig_val], y=[""], orientation='h', marker_color='#1d4ed8',
                                text=[f"{orig_val:.4f}"], textposition='inside', insidetextanchor='middle', showlegend=False
                            ),
                            go.Bar(
                                name='Modified', x=[mod_val], y=[""], orientation='h', marker_color='#059669',
                                text=[f"{mod_val:.4f}"], textposition='inside', insidetextanchor='middle', showlegend=False
                            )
                        ])

                        bar_fig.update_layout(
                            barmode='group',
                            height=45,
                            margin=dict(l=0, r=0, t=0, b=0),
                            title=None,
                            xaxis=dict(
                                showgrid=False,
                                zeroline=True,
                                showticklabels=True,
                                tickfont=dict(size=8),
                                range=[x_range_min, x_range_max],
                            ),
                            yaxis=dict(showticklabels=False),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            transition={'duration': 100},
                        )
                        bar_fig.update_traces(
                            textfont_size=8,
                            cliponaxis=False,
                            width=0.6,
                        )
                        # --- End Mini Dual Bar Chart Figure ---

                        feature_details.append(
                            html.Div(
                                style={'padding': '5px 0', 'border-bottom': '1px dotted #e5e7eb', 'margin-bottom': '5px'},
                                children=[
                                    # Feature Label (Left-aligned)
                                    html.Span(feat_label, style={'font-size': '0.5rem', 'font-weight': '600', 'color': '#374151', 'display': 'block', 'margin-bottom': '2px'}),
                                    # Dual Bar Chart (Full width)
                                    dcc.Graph(
                                        id={'type': 'district-feature-chart', 'index': f"{district}-{feat}"},
                                        figure=bar_fig,
                                        config={'displayModeBar': False},
                                        style={'height': '30px', 'width': '100%'}
                                    )
                                ]
                            )
                        )

            change_df = pd.DataFrame({
                "Change": [change_value],
                "District": [district]
            })
            # app_state = json.loads(app_state_json)
            mini_map=generate_mini_map(change_df,app_state, selected_districts=district)
            # print(mini_map.data)

            # --- Combined District Block ---
            list_elements.append(
                html.Div(
                    style={
                        'padding': '15px',
                        'margin-bottom': '15px',
                        'background-color': '#ffffff',
                        'border': '1px solid #dbeafe',
                        'border-radius': '8px',
                        'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.05)',
                    },
                    children=[
                        # District Title (Header)
                        html.Div(
                            style={'font-weight': '700', 'color': '#1d4ed8', 'font-size': '1.25rem', 'margin-bottom': '10px', 'padding-bottom': '8px', 'border-bottom': '2px solid #93c5fd'},
                            children=district
                        ),

                        # Feature Details Header

                        # # Feature Details List
                        # html.Div(
                        #     style={'max-height': '200px', 'overflowY': 'auto', 'padding-right': '5px'},
                        #     children=feature_details
                        # ) if feature_details else html.P("No features were adjusted for this district, or none are currently selected for intervention.", className='text-xs italic text-gray-500 mt-2 p-2'),

                        # -----------------------------------------------------------------------------------
                        # 🌟 KEY CHANGE: New Flex Container for Side-by-Side Layout
                        # -----------------------------------------------------------------------------------
                        html.Div(
                            style={
                                'display': 'flex',        # Enable Flexbox
                                'flex-direction': 'row',  # Arrange children horizontally
                                'gap': '5px',            # Add space between map and bar chart
                                'margin-top': '15px'      # Space from the feature details above
                            },
                            children=[
                                # 1. Mini Map Container (Left Side)
                                html.Div(
                                    style={'flex': '0.45', 'min-width': '0'}, # Flex: 1 makes it take up half the available space
                                    children=[
                                        dcc.Loading(
                                            id="loading-detail-map",
                                            type="default",
                                            children=dcc.Graph(
                                                id=f'detail-impact-map-chart'+district,
                                                figure=mini_map,
                                                config={
                                                    'displaylogo': False,
                                                    'scrollZoom': True,
                                                    'modeBarButtonsToAdd': ['lasso2d', 'box_select']
                                                },
                                                style={'height': '125px'} # Increased height slightly for better visual
                                            )
                                        )
                                    ]
                                ),

                                # 2. Bar Chart Container (Right Side)
                                html.Div(
                                    style={
                                        'flex': '0.55',
                                        'min-width': '0',
                                        # --- ADDED SCROLLING STYLES ---
                                        'maxHeight': '110px',
                                        'overflowY': 'auto',
                                        'paddingRight': '5px', # Optional: for visual padding near the scrollbar
                                        # -----------------------------
                                    },
                                    children=feature_details
                                ),
                            ]
                        )
                        # -----------------------------------------------------------------------------------
                    ]
                )
            )

        print(f"DEBUG: [CB5] Finished processing. Returning {len(list_elements)} district detail blocks.")
        return list_elements
