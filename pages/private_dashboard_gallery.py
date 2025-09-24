import json
import os
import joblib
import shutil
import geopandas as gpd
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from streamlit.errors import StreamlitValueAssignmentNotAllowedError
from xgboost import XGBRegressor
from uuid import uuid4

# Assuming these imports exist in the user's project
# from pages.home import load_geo
# from pages.public_dashboard_view import load_session_state_from_dir

# --- Helper Functions for JSON File Management ---
DASHBOARD_FILE = "dashboards.json"
BASE_DIR = "published_dashboards"

def get_dashboards_from_json():
    """Reads the dashboard info from the JSON file."""
    if not os.path.exists(DASHBOARD_FILE):
        return []
    with open(DASHBOARD_FILE, "r") as f:
        return json.load(f)

def save_dashboards_to_json(dashboards):
    """Writes the dashboard info to the JSON file."""
    with open(DASHBOARD_FILE, "w") as f:
        json.dump(dashboards, f, indent=4)

def update_dashboard_info_file(new_dashboard_info, action="add"):
    """
    Updates the main dashboards.json file.
    Actions: "add", "delete", "update_privacy".
    """
    dashboards = get_dashboards_from_json()

    if action == "add":
        # Check if a dashboard with the same name already exists
        if any(d['name'] == new_dashboard_info['name'] for d in dashboards):
            st.warning(f"A dashboard with the name '{new_dashboard_info['name']}' already exists. Please choose a different name.")
            return False
        dashboards.append(new_dashboard_info)
    elif action == "delete":
        dashboards = [d for d in dashboards if d['id'] != new_dashboard_info['id']]
    elif action == "update_privacy":
        for d in dashboards:
            if d['id'] == new_dashboard_info['id']:
                d['privacy'] = new_dashboard_info['privacy']
                break

    save_dashboards_to_json(dashboards)
    return True

# --- Main Application Logic ---

def save_session_state_to_dir(dashboard_name, privacy_setting):
    """
    Saves the current Streamlit session state and updates the JSON file.
    """
    # Generate a unique ID for the new dashboard
    dashboard_id = str(uuid4())

    # Create a unique directory based on the ID
    save_path = os.path.join(BASE_DIR, dashboard_id)
    if os.path.exists(save_path):
        st.warning(f"A dashboard folder with the ID '{dashboard_id}' already exists. This is an unexpected error.")
        return False

    os.makedirs(save_path)

    try:
        data_to_save = {}
        for key, value in st.session_state.items():
            # Special handling for non-serializable objects
            if isinstance(value, gpd.GeoDataFrame):
                geo_file_path = os.path.join(save_path, f"{key}.geojson")
                value.to_file(geo_file_path, driver='GeoJSON')
                data_to_save[key] = {"value": geo_file_path, "type": "GeoDataFrame"}
            elif isinstance(value, pd.DataFrame):
                df_file_path = os.path.join(save_path, f"{key}.parquet")
                value.to_parquet(df_file_path)
                data_to_save[key] = {"value": df_file_path, "type": "DataFrame"}
            elif isinstance(value, (RandomForestRegressor, XGBRegressor, LinearRegression, StandardScaler, dict, pd.Series, np.ndarray)):
                file_path = os.path.join(save_path, f"{key}.joblib")
                joblib.dump(value, file_path)
                data_to_save[key] = {"value": file_path, "type": "joblib_file"}
            else:
                try:
                    json.dumps(value)
                    data_to_save[key] = {"value": value, "type": type(value).__name__}
                except TypeError:
                    data_to_save[key] = {"value": str(value), "type": "str"}

        config_file_path = os.path.join(save_path, "config.json")
        with open(config_file_path, "w") as f:
            json.dump(data_to_save, f, indent=4)

        # Update the centralized JSON file with new dashboard info
        new_dashboard_info = {
            "id": dashboard_id,
            "name": dashboard_name,
            "privacy": privacy_setting,
            "path": save_path
        }
        return update_dashboard_info_file(new_dashboard_info, action="add")

    except Exception as e:
        st.error(f"Error saving dashboard: {e}")
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        return False


def delete_dashboard(dashboard_info):
    """Deletes a published dashboard's directory and updates the JSON file."""
    dashboard_path = dashboard_info["path"]
    try:
        if os.path.exists(dashboard_path):
            shutil.rmtree(dashboard_path)
            st.success(f"Dashboard '{dashboard_info['name']}' has been deleted successfully.")
            # Update the centralized JSON file
            update_dashboard_info_file(dashboard_info, action="delete")
            st.rerun()
        else:
            st.warning("Dashboard folder not found, but it will be removed from the list.")
            update_dashboard_info_file(dashboard_info, action="delete")
    except Exception as e:
        st.error(f"Error deleting dashboard: {e}")

def switch_dashboard_privacy(dashboard_info):
    """Switches a dashboard's privacy and updates the JSON file."""
    try:
        new_privacy = "Private" if dashboard_info['privacy'] == "Public" else "Public"
        dashboard_info['privacy'] = new_privacy
        # Update the centralized JSON file
        success = update_dashboard_info_file(dashboard_info, action="update_privacy")
        if success:
            st.success(f"Dashboard '{dashboard_info['name']}' has been switched to '{new_privacy}'.")
            st.rerun()
    except Exception as e:
        st.error(f"Error switching privacy: {e}")


def load_session_state_from_dir(dashboard_path):
    # This is a placeholder for your existing function
    st.info(f"Loading dashboard from {dashboard_path}...")
    # Add your actual loading logic here
    st.session_state.logged_in = True
    st.success("Dashboard Loaded! (Placeholder)")


def run_private_dashboard_gallery():
    """
    A Streamlit view to publish and load dashboards.
    """
    st.title("Publish Dashboard")
    st.markdown("Save your current dashboard configuration for future use or sharing.")

    dashboard_name = st.text_input("Enter a name for your dashboard:", help="This name will be used as the folder name. Avoid special characters.")
    privacy_setting = st.radio("Who can see this dashboard?", ('Private', 'Public'))

    if st.button("Publish Dashboard"):
        if not dashboard_name:
            st.warning("Please enter a name for the dashboard.")
        else:
            with st.spinner("Publishing..."):
                success = save_session_state_to_dir(dashboard_name, privacy_setting)
            if success:
                st.success(f"Dashboard '{dashboard_name}' has been published successfully as '{privacy_setting.lower()}'.")
                st.rerun()
            else:
                st.error("Failed to publish the dashboard.")

    st.markdown("---")
    st.title("Manage Published Dashboards")
    st.markdown("Select a dashboard from the list below to load, delete, or change its privacy.")

    # Get dashboard list from the central JSON file
    all_dashboards = get_dashboards_from_json()

    if all_dashboards:
        # Create a dictionary to map the display name to the full dashboard object
        display_to_dashboard_map = {f"{d['name']} ({d['privacy']})": d for d in all_dashboards}
        display_names = list(display_to_dashboard_map.keys())

        selected_display_name = st.radio("Select a dashboard:", display_names, key="combined_dashboard_selector")

        # Get the full dashboard object from the selected display name
        selected_dashboard = display_to_dashboard_map.get(selected_display_name)

        if selected_dashboard:
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Load Dashboard"):
                    with st.spinner("Loading..."):
                        load_session_state_from_dir(selected_dashboard['path'])
            with col2:
                new_privacy = "private" if selected_dashboard["privacy"] == "Public" else "public"
                if st.button(f"Switch to {new_privacy.capitalize()}"):
                    with st.spinner("Switching..."):
                        switch_dashboard_privacy(selected_dashboard)
            with col3:
                delete_confirmed = st.checkbox("Confirm Deletion", key="delete_confirm")
                if st.button("Delete Dashboard", disabled=not delete_confirmed):
                    with st.spinner("Deleting..."):
                        delete_dashboard(selected_dashboard)
        else:
            st.info("No dashboard selected.")
    else:
        st.info("No published dashboards found.")


