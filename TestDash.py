import socket

import streamlit as st
import streamlit.components.v1 as components
import threading
import time
import json
import numpy as np
from uuid import uuid4

from Dash_Pages.dash_prescriptive_modelling import PYTHON_OBJECT_LOOKUP, run_dash_server_thread_target
# --- IMPORT SHARED STATE AND SERVER LOGIC ---
# We import the global lookup table and the function that runs the Dash server thread.

# NOTE: External dependencies used for initial data setup in st.session_state
from pages.public_dashboard_view import load_first_dashboard_for_testing_automatic
# load_first_dashboard_for_testing_automatic()

# --- 2. DATA PREPARATION LOGIC (Runs in Streamlit's Thread) ---

def find_free_port_in_range(start_port=8600, end_port=8700, default_port=8600):
    """
    Finds a free port within a specified range [start_port, end_port].

    If no free port is found in the range, returns the default_port.
    If the default_port is used and is not free, it may raise an OSError
    when the caller attempts to use it, as this function can't guarantee
    the default_port's availability.
    """
    for port in range(start_port, end_port + 1):
        try:
            # Create a socket using IPv4 and TCP
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Attempt to bind the socket to the port
                s.bind(('', port))
                # If bind succeeds, the port is free; return it
                return port
        except OSError:
            # If bind fails (port is in use or outside allowed range), try the next port
            continue

    # If the loop finishes without finding a free port, return the default port
    return default_port

def prepare_dash_data_from_session_state(required_keys: list) -> tuple[str | None, str | None, str | None]:
    """
    Extracts, serializes, and globally stores necessary data from st.session_state
    for the Dash application running in a separate thread.

    Returns: (df_json, app_state_json, geo_col)
    """


    # Check for all required keys including map data
    map_keys = ["gdf", "GEO_COL"]
    full_required_keys = required_keys + [k for k in map_keys if k not in required_keys]

    missing_keys = [k for k in full_required_keys if k not in st.session_state]
    if missing_keys:
        st.error(f"Missing critical data in st.session_state: {', '.join(missing_keys)}. Please ensure models are trained and data is loaded.")
        return None, None, None

    # 2. Extract DataFrames and primary values
    df = st.session_state["grouped_df"].copy()
    geo_unit_col = st.session_state["geo_unit_col"]
    target_col = st.session_state["target_col"]
    trained_models_session = st.session_state["trained_models"]
    # Retrieve selected_state directly for JSON serialization (no global lookup needed)
    selected_state = st.session_state["selected_state"]
    # --- HANDLE NON-SERIALIZABLE OBJECTS (Store references in PYTHON_OBJECT_LOOKUP) ---

    # a. GeoDataFrame (gdf) and Geo Column Mapping
    gdf_obj = st.session_state["gdf"]
    geo_col_mapping = st.session_state["GEO_COL"]

    # Generate unique ID and store the GeoDataFrame object in the shared lookup table
    gdf_id = f"GDF_{uuid4()}"
    PYTHON_OBJECT_LOOKUP[gdf_id] = gdf_obj
    # Store these key strings globally for use in the calculate_impact placeholder logic
    PYTHON_OBJECT_LOOKUP["geo_unit_col"] = geo_unit_col
    PYTHON_OBJECT_LOOKUP["target_col"] = target_col

    # b. Scaler
    scaler_obj = st.session_state["scaler"]
    scaler_id = f"SCALER_{uuid4()}"
    PYTHON_OBJECT_LOOKUP[scaler_id] = scaler_obj # Store object in shared global state

    # c. Models
    dash_models_refs = {}
    for name, obj in trained_models_session.items():
        # Check if model object exists (can be None if training failed)
        if obj is not None:
            obj_id = f"MODEL_{uuid4()}"
            PYTHON_OBJECT_LOOKUP[obj_id] = obj # Store object in shared global state
            dash_models_refs[name] = obj_id
        else:
            dash_models_refs[name] = None
    if "target_val" not in st.session_state:
        st.session_state["target_val"]=0

    # --- Extract Features and Importances (JSON serializable) ---

    # Safely get optional/complex data
    feature_order = st.session_state.get("final_selected_features", [])
    user_coefs = st.session_state.get("user_coefs", None)

    # Determine significant features from importances
    sig_feats = []
    importances_store = {}
    if "final_feature_importances" in st.session_state:
        # Serialize importances (which are DataFrames)
        importances_dict = st.session_state["final_feature_importances"]
        PYTHON_OBJECT_LOOKUP['final_feature_importances'] = importances_dict

        for model_name, df_importance in importances_dict.items():
            if not df_importance.empty:
                # Store the DataFrame's JSON representation
                importances_store[model_name] = df_importance.to_json(orient='split')
                # Use linear model features as the basis for SIG_FEATS
                if model_name == 'linear':
                    sig_feats = df_importance['Feature'].tolist()

    if not sig_feats and feature_order:
        sig_feats = feature_order

    # Store SIG_FEATS_LIST globally for dynamic slider callback input definition in Dash
    PYTHON_OBJECT_LOOKUP['SIG_FEATS_LIST'] = sig_feats

    # 4. Construct JSON-serializable App State
    app_state = {
        "trained_models": dash_models_refs,
        "final_feature_importances": importances_store,
        "feature_buckets_grouped": st.session_state.get("feature_buckets_grouped", {}),
        "target_direction": st.session_state.get("target_direction", "Increase"),
        "final_positive": st.session_state.get("final_positive", []),
        "final_negative": st.session_state.get("final_negative", []),
        "trained_feature_names": feature_order,
        "user_coefs": user_coefs.tolist() if isinstance(user_coefs, np.ndarray) else None,
        "scaler": scaler_id,
        "target_col": target_col,
        "sig_feats": sig_feats,
        # selected_state is passed directly in the JSON
        "selected_state": selected_state,
        "geo_unit_col": st.session_state["geo_unit_col"],
        "GEO_COL":st.session_state["GEO_COL"],
        "geo_col_mapping":st.session_state["GEO_COL"],
        "gdf_ref": gdf_id # <--- PASS THE UNIQUE ID AS THE REFERENCE
        ,"target_val":st.session_state["target_val"]
    }
    # print()
    # st.info(app_state)
    # 5. Serialize DataFrames
    df_json = df.to_json(date_format='iso', orient='split')
    app_state_json = json.dumps(app_state)

    return df_json, app_state_json, geo_unit_col

# --- 10. STREAMLIT EMBEDDING UI (Main Application Entry Point) ---
def  UseDashPrescriptiveAnalysis():
    st.set_page_config(layout="wide", page_title="Combined Differential Impact App")
    # Required keys for the Dash app to function
    REQUIRED_KEYS = ["selected_state", "gdf", "GEO_COL", "grouped_df", "geo_unit_col", "target_col", "trained_models", "scaler",
                     "final_feature_importances", "target_direction", "feature_buckets_grouped"]

    # Step 1: Prepare data and start thread if ready
    df_json, app_state_json, geo_col = prepare_dash_data_from_session_state(REQUIRED_KEYS)

    if df_json is None:
        st.stop()

    # Step 2: Configure and Start Dash server thread
    DYNAMIC_PORT = find_free_port_in_range(default_port=8050)
    DASH_APP_URL = f"http://lki-ssm.iiitb.ac.in/{DYNAMIC_PORT}/"

    if 'dash_server_thread' not in st.__dict__:
        st.dash_server_thread = threading.Thread(
            target=run_dash_server_thread_target,
            args=(df_json, app_state_json, geo_col, DYNAMIC_PORT),
            daemon=True
        )
        st.dash_server_thread.start()
        time.sleep(2) # Give the server a moment to spin up
        print(f"Dash Server successfully initialized in thread on {DASH_APP_URL}")


    st.markdown("""
        <style>
        .stApp { padding-top: 2rem; }
        .main-header {
            text-align: center; color: #0c4a6e; font-size: 2.25rem;
            font-weight: 700; margin-bottom: 0.5rem;
        }
        .status-message {
            text-align: center; color: #374151; font-size: 1rem;
            margin-bottom: 2rem;
        }
        .iframe-container {
            border: 4px solid #3b82f6; border-radius: 1rem;
            overflow: hidden; box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.2);
            padding: 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # st.markdown('<div class="main-header">Full Streamlit and Dash Differential Impact Analysis</div>', unsafe_allow_html=True)
    # st.markdown(f'<div class="status-message">Dash application running in a background thread on {DASH_APP_URL} and embedded below.</div>', unsafe_allow_html=True)

    IFRAME_HEIGHT = 4600

    iframe_html = f"""
    <div class="iframe-container" style="height: {IFRAME_HEIGHT}px;">
        <iframe
            src=f"http://127.0.0.1:{DYNAMIC_PORT}"
            style="width: 100%; height: 100%; border: none; padding: 0;"
            title="Embedded Dash Application"
        ></iframe>
    </div>
    """
    st.session_state["Dash_URL"]=DASH_APP_URL
    # Create a container with a visible border
    with st.container(border=True):
        # Use a bold label and an emoji to increase visual prominence
        st.page_link(
            DASH_APP_URL,
            label="🚀 **Open this Dashboard on a new Page** 🚀",
            icon="👉" # Add a directional icon
        )


    st.info("Note that this page is embedding using a third party server it is preferred to view this page in a new tab")
    # st.link_button()
    components.html(iframe_html, height=IFRAME_HEIGHT + 30, scrolling=True)

    st.markdown("---")
    return DASH_APP_URL
    # st.code({DASH_APP_URL}", language='python')
