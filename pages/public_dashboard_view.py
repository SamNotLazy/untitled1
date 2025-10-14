import json
import os
import time

import joblib
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd

# These imports are kept for deserialization of saved objects
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from streamlit.errors import StreamlitValueAssignmentNotAllowedError
from xgboost import XGBRegressor

from pages.home import load_geo

# This is a placeholder for the user's load_geo function
# def load_geo(filepath):
#     """
#     A placeholder function to load a GeoDataFrame.
#     Assumes a GeoJSON file and a geometry column.
#     """
#     try:
#         gdf = gpd.read_file(filepath)
#         # Assuming the geometry column is named 'geometry'
#         geo_col = 'geometry'
#         return gdf, geo_col
#     except Exception as e:
#         st.error(f"Error loading GeoJSON file: {e}")
#         return None, None

# --- Helper Functions for JSON File Management ---
DASHBOARD_FILE = "dashboards.json"
BASE_DIR = "published_dashboards"

def get_dashboards_from_json():
    """Reads the dashboard info from the JSON file."""
    if not os.path.exists(DASHBOARD_FILE):
        return []
    with open(DASHBOARD_FILE, "r") as f:
        return json.load(f)

def load_session_state_from_dir(dashboard_path):
    """
    Loads the Streamlit session state from a published dashboard's directory
    and sets a session state flag for successful loading.
    """
    # Initialize the flag to False before attempting to load
    st.session_state.dashboard_loaded = False
    config_file_path = os.path.join(dashboard_path, "config.json")
    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, "r") as f:
                loaded_data = json.load(f)

            for key, item_data in loaded_data.items():
                value = item_data.get("value")
                type_name = item_data.get("type")
                if key in ["editor_linear", "multi_uploader",
                           'FormSubmitter:agg_chooser_form-Apply aggregation & build grouped view',
                           'FormSubmitter:bucket_add_form-Add Bucket', 'FormSubmitter:bucket_remove_form-Remove Selected',
                           "combined_dashboard_selector","public_sub_page","logged_in_main_section","logged_in_sub_page","pre_login_page"]:
                    continue
                if type_name == "GeoDataFrame":
                    # Load GeoDataFrame from the saved geojson file
                    if os.path.exists(value):
                        st.session_state[key], st.session_state["GEO_COL"] = load_geo(value)
                    else:
                        st.error(f"GeoDataFrame file not found: {value}")
                        return
                elif type_name == "DataFrame":
                    # Load DataFrame from the saved parquet file
                    if os.path.exists(value):
                        st.session_state[key] = pd.read_parquet(value)
                elif type_name == "joblib_file":
                    # Load the serialized object from the file
                    if os.path.exists(value):
                        st.session_state[key] = joblib.load(value)
                else:
                    # For other types, directly assign the value
                    st.session_state[key] = value

            # If all items are loaded successfully, set the flag to True
            st.session_state.dashboard_loaded = True
            message = f"Dashboard '{os.path.basename(dashboard_path)}' loaded successfully! ✅"

            # 2. Create a placeholder in the app
            # The placeholder is an empty container where your message will go.
            placeholder = st.empty()

            # 3. Display the message inside the placeholder using st.success()
            placeholder.success(message)

            # 4. Wait for the desired duration (e.g., 3 seconds)
            time.sleep(1)

            # 5. Clear the placeholder to make the message disappear
            placeholder.empty()
        except Exception as e:
            st.error(f"Error loading dashboard: {e}")
            st.session_state.dashboard_loaded = False
    else:
        st.error("Dashboard configuration file not found.")
        st.session_state.dashboard_loaded = False


def run_public_dashboard_view():
    """
    A Streamlit view to list and load only public dashboards using buttons.
    """
    st.title("Load Public Dashboards")
    st.markdown("Select a publicly saved dashboard from the list below to load its configuration.")

    # Get all dashboards from the central JSON file
    all_dashboards = get_dashboards_from_json()

    # Filter for only public dashboards
    public_dashboards = [d for d in all_dashboards if d.get('privacy') == 'Public']

    if public_dashboards:
        # Create a button for each public dashboard
        for dashboard in public_dashboards:
            col1, col2 = st.columns([0.7, 0.3])

            with col1:
                st.markdown(f"**{dashboard['name']}**")

            with col2:
                if st.button("Load", key=f"load_button_{dashboard['id']}"):
                    with st.spinner("Loading..."):
                        st.session_state["loaded_dashboard_name"] = dashboard['name']

                        load_session_state_from_dir(dashboard['path'])
                        st.session_state.logged_in = False
    else:
        st.info("No public dashboards found.")

def load_first_dashboard_for_testing_automatic():
    """
    Automatically attempts to load the first available dashboard
    and displays the results and loaded session state keys.
    """

    all_dashboards = get_dashboards_from_json()
    loaded_name = st.session_state.get("loaded_dashboard_name")

    if not all_dashboards:
        st.warning(f"❌ No published dashboards found in **{DASHBOARD_FILE}**.")
        st.info("Please publish a dashboard using your main app first.")
        return

    first_dashboard = all_dashboards[0]
    dashboard_name = first_dashboard.get("name", "Unknown Dashboard")
    dashboard_path = first_dashboard.get("path")



    # --- CORE AUTO-LOAD LOGIC ---
    if loaded_name != dashboard_name:
        # Only load if it hasn't been loaded yet in this session
        with st.spinner(f"**AUTO-LOADING** session state from {dashboard_path}..."):
            try:
                # 1. Load the data into st.session_state
                load_session_state_from_dir(dashboard_path)
                st.session_state["loaded_dashboard_name"] = dashboard_name

                # st.success(f"✅ Automatically loaded dashboard **{dashboard_name}**!")
                # st.rerun() # Typically not needed here unless the loaded state dramatically changes the layout above this point.

            except Exception as e:
                st.error(f"❌ Failed to load dashboard **{dashboard_name}**.")
                st.exception(e)
    else:
        st.success(f"✅ Dashboard **{dashboard_name}** is already loaded.")



# This block ensures the script can be run directly as a Streamlit app
if __name__ == "__main__":
    run_public_dashboard_view()
