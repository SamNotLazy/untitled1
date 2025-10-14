import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import json
import math
from collections import defaultdict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd # IMPORTANT: Uncomment if loading a GeoJSON or GDF from a file path

# --------------------------------------------------------------------------------
# 1. Helper Functions
# --------------------------------------------------------------------------------

def _calculate_bounds(coordinates):
    """
    Calculates the minimum and maximum latitude and longitude for a given GeoJSON coordinate structure.
    Returns (min_lon, min_lat, max_lon, max_lat).
    """
    if not coordinates:
        return None

    # Flatten coordinates into a list of (lon, lat) tuples
    def flatten(coords):
        flat_coords = []
        for coord in coords:
            # Handle nested structures
            if isinstance(coord[0], (int, float)): # (lon, lat) pair
                flat_coords.append(coord)
            else: # Nested structure (LineString in Polygon, or Multi*)
                flat_coords.extend(flatten(coord))
        return flat_coords

    # Note: GeoJSON 'coordinates' can be deeply nested depending on type (e.g., MultiPolygon)
    # The existing flatten needs to handle the top level correctly.
    # For a standard single-geometry, we pass the coordinates array.

    # We pass the coordinates as a list of lists/tuples to the recursive flatten function
    flat_coords = flatten(coordinates)

    if not flat_coords:
        return None

    min_lon = min(c[0] for c in flat_coords)
    min_lat = min(c[1] for c in flat_coords)
    max_lon = max(c[0] for c in flat_coords)
    max_lat = max(c[1] for c in flat_coords)

    return min_lon, min_lat, max_lon, max_lat

def calculate_initial_fit(geometry):
    """
    Calculates the center and an approximate zoom level for a given GeoJSON geometry
    using a simple heuristic based on the degree extent.
    """
    if not geometry or not geometry.get('coordinates'):
        return {'lat': 0, 'lon': 0, 'zoom': 2}

    try:
        bounds = _calculate_bounds(geometry['coordinates'])
        if not bounds:
            return {'lat': 0, 'lon': 0, 'zoom': 2}

        min_lon, min_lat, max_lon, max_lat = bounds

        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0

        lon_diff = max_lon - min_lon
        lat_diff = max_lat - min_lat
        max_diff = max(lon_diff, lat_diff)

        if max_diff == 0:
            zoom = 15.0 # For single points
        else:
            # Simple log approximation for zoom
            zoom = min(18.0, 1.0 + math.log2(360.0 / max_diff))

        # Apply a small buffer
        zoom = max(1.0, zoom - 0.5)

        return {
            'lat': round(center_lat, 6),
            'lon': round(center_lon, 6),
            'zoom': round(zoom, 2)
        }
    except Exception as e:
        print(f"Error calculating fit for geometry: {e}")
        return {'lat': 0, 'lon': 0, 'zoom': 2}

# --------------------------------------------------------------------------------
# 2. Dash Setup and Layout
# --------------------------------------------------------------------------------

# --- HARDCODED FILE LOADING (MOCK DATA FOR DEMO) ---

GEOJSON_FILE_PATH = "States/UTTAR PRADESH/UTTAR PRADESH_DISTRICTS_Old.geojson"

def create_mock_geojson():
    """Creates a basic mock GeoJSON for the state of Uttar Pradesh if loading fails."""
    mock_feature_1 = {
        "type": "Feature",
        "properties": {
            "dtname": "Mock District 1",
            "NAME": "Mock District 1",
            "id": "mock1"
        },
        "geometry": {
            "type": "Polygon",
            # A rough box representing UP's location (23N to 31N, 77E to 84E)
            "coordinates": [[[[77, 23], [80, 23], [80, 27], [77, 27], [77, 23]]]]
        }
    }
    mock_feature_2 = {
        "type": "Feature",
        "properties": {
            "dtname": "Mock District 2",
            "NAME": "Mock District 2",
            "id": "mock2"
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[[80, 27], [84, 27], [84, 31], [80, 31], [80, 27]]]]
        }
    }

    features = [mock_feature_1, mock_feature_2]

    # Calculate fit for individual features and set properties
    for index, feature in enumerate(features):
        fit = calculate_initial_fit(feature['geometry'])
        feature['properties']['_optimal_zoom'] = fit['zoom']
        feature['properties']['_center_lat'] = fit['lat']
        feature['properties']['_center_lon'] = fit['lon']
        feature['properties']['_feature_id'] = feature['properties']['id']
        feature['properties']['_feature_name'] = feature['properties']['dtname']

    # Calculate whole state fit (based on the combined bounds of mock data)
    all_coords = [
        [77, 23], [80, 23], [80, 27], [77, 27],
        [80, 27], [84, 27], [84, 31], [80, 31]
    ]

    min_lon = min(c[0] for c in all_coords)
    min_lat = min(c[1] for c in all_coords)
    max_lon = max(c[0] for c in all_coords)
    max_lat = max(c[1] for c in all_coords)

    lon_diff = max_lon - min_lon
    lat_diff = max_lat - min_lat
    max_diff = max(lon_diff, lat_diff)

    state_zoom = min(18.0, 1.0 + math.log2(360.0 / max_diff)) - 0.5

    whole_state_fit = {
        '_center_lat': round((min_lat + max_lat) / 2.0, 6),
        '_center_lon': round((min_lon + max_lon) / 2.0, 6),
        '_optimal_zoom': round(state_zoom, 2),
        '_feature_name': "All Districts/Whole State (Mock)"
    }

    dropdown_options = [{'label': f"{index + 1}. {f['properties']['dtname']}", 'value': index} for index, f in enumerate(features)]

    return {
               'type': 'FeatureCollection',
               'features': features,
               '_whole_state_fit': whole_state_fit # Store special fit data
           }, "Warning: Using mock data. Please ensure geopandas is installed and the GeoJSON path is correct.", dropdown_options,

def load_initial_geojson_data():
    """
    Loads GeoJSON data from the specified path and calculates initial fit for
    individual features and the entire collection, falling back to mock data if needed.
    """
    dropdown_options = []

    try:
        # 1. Load data and calculate whole state fit using GeoPandas
        gdf = gpd.read_file(GEOJSON_FILE_PATH)
        data = gdf.__geo_interface__
        status = f"Success! Loaded {len(data['features'])} features from {GEOJSON_FILE_PATH} using GeoPandas."

        # Calculate overall bounds and fit for the WHOLE STATE/COLLECTION
        state_bounds = gdf.total_bounds # (minx, miny, maxx, maxy) -> (min_lon, min_lat, max_lon, max_lat)
        state_lon_diff = state_bounds[2] - state_bounds[0]
        state_lat_diff = state_bounds[3] - state_bounds[1]
        state_max_diff = max(state_lon_diff, state_lat_diff)

        state_center_lat = (state_bounds[1] + state_bounds[3]) / 2.0
        state_center_lon = (state_bounds[0] + state_bounds[2]) / 2.0

        # Calculate zoom for the whole state (using the same heuristic)
        state_zoom = 15.0 if state_max_diff == 0 else min(18.0, 1.0 + math.log2(360.0 / state_max_diff))
        state_zoom = max(1.0, state_zoom - 0.5)

        whole_state_fit = {
            '_center_lat': round(state_center_lat, 6),
            '_center_lon': round(state_center_lon, 6),
            '_optimal_zoom': round(state_zoom, 2),
            '_feature_name': "All Districts/Whole State"
        }

    except Exception as e:
        print(f"Error loading real GeoJSON file: {e}. Falling back to mock data.")
        # 2. Fallback to mock data logic
        data, status, temp_options = create_mock_geojson()
        dropdown_options.extend(temp_options)
        whole_state_fit = data.pop('_whole_state_fit') # Retrieve special fit data

    all_features = data['features']

    # 3. Prepend the 'Whole State' option (-1 index)
    dropdown_options.insert(0, {'label': f"0. {whole_state_fit['_feature_name']}", 'value': -1})

    # 4. Process individual features and calculate individual fit parameters
    for index, feature in enumerate(all_features):
        if 'properties' not in feature: feature['properties'] = {}

        feature_id = feature['properties'].get('id', str(index))
        feature['properties']['_feature_id'] = feature_id
        name = feature['properties'].get('dtname', feature['properties'].get('NAME', f"Feature {index + 1}"))
        feature['properties']['_feature_name'] = name

        if 'geometry' in feature and feature['geometry'] and '_optimal_zoom' not in feature['properties']:
            # Only calculate if not already done by mock data logic
            fit = calculate_initial_fit(feature['geometry'])
            feature['properties']['_optimal_zoom'] = fit['zoom']
            feature['properties']['_center_lat'] = fit['lat']
            feature['properties']['_center_lon'] = fit['lon']
        elif '_optimal_zoom' not in feature['properties']:
            feature['properties']['_optimal_zoom'] = 2
            feature['properties']['_center_lat'] = 0
            feature['properties']['_center_lon'] = 0

        # Add individual feature option if not already added by mock data logic
        if not any(opt['value'] == index for opt in dropdown_options):
            dropdown_options.append({'label': f"{index + 1}. {name}", 'value': index})

    # 5. Store the Whole State Fit in the data structure under a special key
    data['_whole_state_fit'] = whole_state_fit

    # 6. Set initial value to Whole State
    return data, status, dropdown_options, -1


# Load data immediately upon script execution
INITIAL_DATA, INITIAL_STATUS, INITIAL_OPTIONS, INITIAL_VALUE = load_initial_geojson_data()


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[
    # Using Tailwind CSS classes for styling
    "https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"
])

# Function to create a base Plotly figure (minimal, white background)
def create_base_figure():
    return go.Figure(
        layout=dict(
            mapbox_style="white-bg",
            mapbox_zoom=INITIAL_DATA.get('_whole_state_fit', {}).get('_optimal_zoom', 2),
            mapbox_center={
                "lat": INITIAL_DATA.get('_whole_state_fit', {}).get('_center_lat', 20),
                "lon": INITIAL_DATA.get('_whole_state_fit', {}).get('_center_lon', 0)
            },
            margin={"r":0,"t":0,"l":0,"b":0},
            uirevision=True,
            coloraxis_showscale=False
        )
    )

# --- CLIENTSIDE CALLBACK REGISTRATION ---
# Registers the clientside callback to track window width and store it.
app.clientside_callback(
    """
    function(dummy_input) {
        function update_width() {
            var width = window.innerWidth;
            // Set 'data' property of the dcc.Store
            window.dash_clientside.set_props('window-width-store', {data: width});
        }
        // Poll for width updates every second
        setInterval(update_width, 1000); 
        update_width(); // Initial call
        return window.dash_clientside.no_update;
    }
    """,
    Output('dummy-div-for-clientside-callback', 'children'),
    Input('dummy-div-for-clientside-callback', 'children')
)

app.layout = html.Div(className="bg-gray-50 p-4 sm:p-8 font-sans antialiased min-h-screen", children=[

    # Hidden components
    dcc.Store(id='geojson-store', data=INITIAL_DATA),
    dcc.Download(id="download-geojson"),
    # New components for clientside width tracking
    dcc.Store(id='window-width-store', data=None),
    html.Div(id='dummy-div-for-clientside-callback', style={'display': 'none'}),

    # Removed html.Script as clientside callback handles width tracking

    # Removed max-w-6xl so the content utilizes the full screen width
    html.Div(className="w-full mx-auto px-4", children=[

        # Header with dynamic width display
        html.Div(className="flex justify-between items-center mb-6 py-4 border-b border-gray-200", children=[
            html.H1("District Map Fit Optimizer", className="text-3xl font-extrabold text-gray-900"),
            html.P(id="available-width-display", className="text-xl font-mono font-semibold text-violet-600 bg-violet-50 p-2 rounded-lg", children="Viewport Width: N/A")
        ]),

        # Main 3-Column Grid Layout - Using md:grid-cols-3 for earlier 1/3 split
        html.Div(className="grid grid-cols-1 md:grid-cols-3 gap-6", children=[

            # Column 1: Controls and List (1/3rd width from md breakpoint up)
            html.Div(className="md:col-span-1 space-y-6", children=[

                # Data Status Card
                html.Div(className="bg-white p-5 rounded-xl shadow-lg border border-gray-100", children=[
                    html.H2("Data Source Status", className="text-xl font-semibold text-gray-700 mb-3"),
                    html.P(id="file-status", className="mt-2 text-sm text-gray-500", children=INITIAL_STATUS)
                ]),

                # District List Card
                html.Div(className="bg-white p-5 rounded-xl shadow-lg border border-gray-100", children=[
                    html.H2("1. Select Feature", className="text-xl font-semibold text-gray-700 mb-3"),
                    dcc.Dropdown(
                        id='feature-dropdown',
                        options=INITIAL_OPTIONS,
                        value=INITIAL_VALUE, # Default to -1 (Whole State)
                        placeholder="Select a feature...",
                        className="rounded-lg shadow-sm"
                    ),
                    html.P(id="current-feature-display", className="mt-3 text-md font-medium text-gray-700", children="No feature selected.")
                ])
            ]),

            # Column 2: Map View Card (1/3rd width from md breakpoint up)
            html.Div(className="md:col-span-1 space-y-6", children=[
                # Map View Card
                html.Div(className="bg-white p-5 rounded-xl shadow-lg border border-gray-100", children=[
                    html.H2("Map Preview", className="text-xl font-semibold text-gray-700 mb-3"),

                    # === INPUTS START (Width is now in Pixels) ===
                    html.Div(className="flex space-x-4 mb-4", children=[
                        # Height Input (Pixels)
                        html.Div(className="w-1/2", children=[
                            html.Label("Map Height (px):", className="text-sm font-medium text-gray-600"),
                            dcc.Input(
                                id='map-height-input',
                                type='number',
                                value=450, # Default height
                                min=100,
                                max=1000,
                                step=1,
                                className="w-full mt-1 p-2 border border-gray-300 rounded-lg focus:ring-violet-500 focus:border-violet-500"
                            )
                        ]),
                        # Width Input (Pixels)
                        html.Div(className="w-1/2", children=[
                            html.Label("Map Width (px):", className="text-sm font-medium text-gray-600"),
                            dcc.Input(
                                id='map-width-input',
                                type='number',
                                value=450, # Default width in pixels
                                min=100,
                                max=1000,
                                step=10,
                                className="w-full mt-1 p-2 border border-gray-300 rounded-lg focus:ring-violet-500 focus:border-violet-500"
                            )
                        ])
                    ]),
                    # === INPUTS END ===

                    dcc.Graph(
                        id="map-graph",
                        figure=create_base_figure(),
                        config={'displayModeBar': False},
                        style={'height': '450px', 'width': '100%', 'borderRadius': '0.5rem', 'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1)'}
                    ),
                ]),
            ]),

            # Column 3: Adjustment and Export Cards (1/3rd width from md breakpoint up)
            html.Div(className="md:col-span-1 space-y-6", children=[
                # Adjustment Card
                html.Div(id="adjustment-card", className="bg-white p-5 rounded-xl shadow-lg border border-gray-100 opacity-100 pointer-events-auto", children=[
                    html.H2("2. Fine-Tune Parameters", className="text-xl font-semibold text-gray-700 mb-3"),

                    # Zoom Slider
                    html.Div(className="flex items-center space-x-4 mb-4", children=[
                        html.Label("Zoom Level:", className="w-32 text-sm font-medium text-gray-700"),
                        dcc.Slider(
                            id='zoom-slider', min=1, max=18, step=0.5,
                            value=INITIAL_DATA.get('_whole_state_fit', {}).get('_optimal_zoom', 10),
                            marks={i: str(i) for i in range(1, 19, 2)},
                            className="w-full h-2 bg-violet-100 rounded-lg cursor-pointer range-lg"
                        ),
                        html.Span(id="zoom-value", className="text-sm font-semibold text-violet-700 w-12 text-right", children="10.0")
                    ]),

                    # Latitude Slider
                    html.Div(className="flex items-center space-x-4 mb-4", children=[
                        html.Label("Center Latitude:", className="w-32 text-sm font-medium text-gray-700"),
                        dcc.Slider(
                            id='lat-slider', min=-90, max=90, step=0.001,
                            value=INITIAL_DATA.get('_whole_state_fit', {}).get('_center_lat', 0),
                            marks={-90: '-90', 0: '0', 90: '90'},
                            className="w-full h-2 bg-blue-100 rounded-lg cursor-pointer range-lg"
                        ),
                        html.Span(id="lat-value", className="text-sm font-semibold text-blue-700 w-12 text-right", children="0.000")
                    ]),

                    # Longitude Slider
                    html.Div(className="flex items-center space-x-4", children=[
                        html.Label("Center Longitude:", className="w-32 text-sm font-medium text-gray-700"),
                        dcc.Slider(
                            id='lon-slider', min=-180, max=180, step=0.001,
                            value=INITIAL_DATA.get('_whole_state_fit', {}).get('_center_lon', 0),
                            marks={-180: '-180', 0: '0', 180: '180'},
                            className="w-full h-2 bg-red-100 rounded-lg cursor-pointer range-lg"
                        ),
                        html.Span(id="lon-value", className="text-sm font-semibold text-red-700 w-12 text-right", children="0.000")
                    ])
                ]),

                # Export Card
                html.Div(className="bg-white p-5 rounded-xl shadow-lg border border-gray-100 flex justify-end", children=[
                    html.Button("3. Export GeoJSON (with New Params)",
                                id='export-button',
                                className="px-6 py-2 bg-green-500 text-white font-semibold rounded-full shadow-md hover:bg-green-600 transition duration-150 disabled:opacity-50",
                                disabled=(INITIAL_DATA is None),
                                n_clicks=0
                                )
                ])
            ])
        ])
    ])
])

# --------------------------------------------------------------------------------
# 3. Callbacks (Interactivity)
# --------------------------------------------------------------------------------

# 3.1 Feature Selection and Map/Slider Update
@app.callback(
    [Output('map-graph', 'figure'),
     Output('lat-slider', 'value'),
     Output('lon-slider', 'value'),
     Output('zoom-slider', 'value'),
     Output('adjustment-card', 'className'),
     Output('current-feature-display', 'children')],
    [Input('feature-dropdown', 'value')],
    [State('geojson-store', 'data')]
)
def select_feature(selected_index, data):
    # Determine which feature to use, or if it's the whole collection
    if data is None or selected_index is None:
        return create_base_figure(), 0, 0, 10, \
               "bg-white p-5 rounded-xl shadow-lg border border-gray-100 opacity-50 pointer-events-none", \
               "No feature selected."

    # --- WHOLE STATE LOGIC (selected_index == -1) ---
    if selected_index == -1:
        fit = data.get('_whole_state_fit', {'_center_lat': 0, '_center_lon': 0, '_optimal_zoom': 2, '_feature_name': "All Districts/Whole State"})

        lat = fit['_center_lat']
        lon = fit['_center_lon']
        zoom = fit['_optimal_zoom']
        name = fit['_feature_name']

        # Plotly setup for all features
        geo_data = {
            'type': 'FeatureCollection',
            'features': data['features'] # Use all features
        }

        # Create a dummy DataFrame where IDs correspond to ALL features
        feature_ids = [f['properties']['_feature_id'] for f in data['features']]
        dummy_df = pd.DataFrame({'id': feature_ids, 'value': 1})
        feature_id_key = "properties._feature_id"

    # --- INDIVIDUAL FEATURE LOGIC (selected_index >= 0) ---
    else:
        try:
            feature = data['features'][selected_index]
        except IndexError:
            # Handle case where index might be out of bounds if data changed
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, \
                   "bg-white p-5 rounded-xl shadow-lg border border-gray-100 opacity-100 pointer-events-auto", \
                   "Error: Index out of bounds."

        props = feature['properties']

        lat = props.get('_center_lat', 0)
        lon = props.get('_center_lon', 0)
        zoom = props.get('_optimal_zoom', 2)
        name = props.get('_feature_name', f"Feature {selected_index + 1}")

        # Plotly setup: Use a dummy DataFrame to draw only the selected GeoJSON
        feature_id = props.get('_feature_id', str(selected_index))
        dummy_df = pd.DataFrame([{'id': feature_id, 'value': 1}])

        geo_data = {
            'type': 'FeatureCollection',
            'features': [feature]
        } if feature.get('geometry') else {'type': 'FeatureCollection', 'features': []}

        feature_id_key = "properties._feature_id"


    # Common Plotting Section
    fig = px.choropleth_mapbox(
        dummy_df,
        geojson=geo_data,
        locations='id',
        featureidkey=feature_id_key,
        color='value',
        color_continuous_scale=[(0, '#8b5cf6'), (1, '#8b5cf6')], # Set single color (Violet)
        range_color=(0, 1),
    )

    # Configure map layout for pure white background and centering
    fig.update_layout(
        showlegend=False,
        margin={"r":0,"t":0,"l":0,"b":0},
        mapbox_style="white-bg", # Pure White map background
        mapbox_center={"lat": lat, "lon": lon},
        mapbox_zoom=zoom,
        uirevision=selected_index, # Re-render map completely when index changes
        coloraxis_showscale=False
    )

    # Style the trace (fill and border)
    fig.update_traces(
        marker_line_width=2 if selected_index == -1 else 3,
        marker_line_color='#4c1d95', # Darker violet border
        marker_opacity=0.7
    )

    active_class = "bg-white p-5 rounded-xl shadow-lg border border-gray-100 opacity-100 pointer-events-auto"

    return fig, lat, lon, zoom, active_class, f"Editing: {name}"

# 3.2 Slider Adjustment and Data Update
@app.callback(
    [Output('geojson-store', 'data', allow_duplicate=True),
     Output('lat-value', 'children'),
     Output('lon-value', 'children'),
     Output('zoom-value', 'children'),
     Output('map-graph', 'figure', allow_duplicate=True)],
    [Input('lat-slider', 'value'),
     Input('lon-slider', 'value'),
     Input('zoom-slider', 'value')],
    [State('feature-dropdown', 'value'),
     State('geojson-store', 'data'),
     State('map-graph', 'figure')],
    prevent_initial_call=True
)
def update_feature_parameters(new_lat, new_lon, new_zoom, selected_index, data, current_figure):
    if data is None or selected_index is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # 1. Update data store
    if selected_index == -1:
        # Update whole state fit
        fit = data.get('_whole_state_fit', {})
        fit['_center_lat'] = new_lat
        fit['_center_lon'] = new_lon
        fit['_optimal_zoom'] = new_zoom
        data['_whole_state_fit'] = fit
    else:
        # Update individual feature parameters
        feature = data['features'][selected_index]
        feature['properties']['_center_lat'] = new_lat
        feature['properties']['_center_lon'] = new_lon
        feature['properties']['_optimal_zoom'] = new_zoom


    # 2. Update value displays
    lat_text = f"{new_lat:.3f}"
    lon_text = f"{new_lon:.3f}"
    zoom_text = f"{new_zoom:.2f}"

    # 3. Update map display (Plotly Figure)
    fig = go.Figure(current_figure)

    # Update map center and zoom in the layout
    fig.update_layout(
        mapbox_center={"lat": new_lat, "lon": new_lon},
        mapbox_zoom=new_zoom,
    )

    return data, lat_text, lon_text, zoom_text, fig

# 3.3 Export Data
@app.callback(
    Output("download-geojson", "data"),
    [Input("export-button", "n_clicks")],
    [State('geojson-store', 'data')],
    prevent_initial_call=True
)
def export_data(n_clicks, data):
    if data is None:
        return dash.no_update

    # Create a copy of the data dictionary to clean up special keys before export
    export_data = data.copy()
    if '_whole_state_fit' in export_data:
        export_data.pop('_whole_state_fit')

    # Create the file content
    geojson_string = json.dumps(export_data, indent=2)

    return dcc.send_bytes(geojson_string.encode('utf-8'), "optimized_districts.geojson")

# 3.4 Map Dimension Update
@app.callback(
    Output('map-graph', 'style'),
    [Input('map-height-input', 'value'),
     Input('map-width-input', 'value')]
)
def update_map_style(height_px, width_px):
    """Updates the dcc.Graph style based on user input for height (px) and width (px)."""
    # Base styles to maintain appearance
    style = {
        'borderRadius': '0.5rem',
        'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
    }

    # Apply height if valid
    if height_px is not None and height_px > 0:
        style['height'] = f"{height_px}px"
    else:
        style['height'] = '450px' # Fallback to default

    # Apply width if valid (in pixels)
    if width_px is not None and width_px > 0:
        style['width'] = f"{width_px}px"
    else:
        style['width'] = '100%' # Fallback to default (full column width)

    return style

# 3.5 Display Available Width (Python Callback reacting to clientside store)
@app.callback(
    Output("available-width-display", 'children'),
    [Input('window-width-store', 'data')]
)
def update_width_display(current_width):
    """Updates the display element based on the value in the dcc.Store."""
    if current_width is None:
        return "Viewport Width: N/A"
    return f"Viewport Width: {current_width}px"


if __name__ == '__main__':
    # Use host='0.0.0.0' for external access/deployment
    app.run(debug=True, host='0.0.0.0', port=8080)
