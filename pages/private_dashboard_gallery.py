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

from pages.home import load_geo


def save_session_state_to_dir(dashboard_name, privacy_setting, base_dir="published_dashboards"):
    """
    Saves the current Streamlit session state to a dedicated directory for publishing.

    This function serializes the session state and any non-serializable objects
    (models, dataframes, etc.) into a new directory.
    """
    # Create the base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Determine the save path based on privacy setting
    save_path = os.path.join(base_dir, privacy_setting, dashboard_name)
    if os.path.exists(save_path):
        st.warning(f"A dashboard with the name '{dashboard_name}' already exists. Please choose a different name.")
        return False

    # Create the directory for the new dashboard
    os.makedirs(save_path)

    try:
        data_to_save = {}
        for key, value in st.session_state.items():
            # Special handling for non-serializable objects
            if isinstance(value, gpd.GeoDataFrame):
                # Save GeoDataFrame as a geojson file
                geo_file_path = os.path.join(save_path, f"{key}.geojson")
                value.to_file(geo_file_path, driver='GeoJSON')
                data_to_save[key] = {
                    "value": geo_file_path,
                    "type": "GeoDataFrame"
                }
            elif isinstance(value, pd.DataFrame):
                # Save DataFrame to a parquet file for efficiency
                df_file_path = os.path.join(save_path, f"{key}.parquet")
                value.to_parquet(df_file_path)
                data_to_save[key] = {
                    "value": df_file_path,
                    "type": "DataFrame"
                }
            # Handle non-serializable models and other objects with joblib
            elif isinstance(value, (RandomForestRegressor, XGBRegressor, LinearRegression, StandardScaler, dict, pd.Series, np.ndarray)):
                file_path = os.path.join(save_path, f"{key}.joblib")
                joblib.dump(value, file_path)
                data_to_save[key] = {
                    "value": file_path,
                    "type": "joblib_file"
                }
            else:
                try:
                    # Attempt to serialize the value to check if it's JSON serializable
                    json.dumps(value)
                    data_to_save[key] = {
                        "value": value,
                        "type": type(value).__name__
                    }
                except TypeError:
                    # st.warning(f"Non-serializable object found for key '{key}'. Saving as a string representation.")
                    data_to_save[key] = {
                        "value": str(value),
                        "type": "str"
                    }

        # Save the main configuration file
        config_file_path = os.path.join(save_path, "config.json")
        with open(config_file_path, "w") as f:
            json.dump(data_to_save, f, indent=4)

        return True

    except Exception as e:
        st.error(f"Error saving dashboard: {e}")
        # Clean up the directory if an error occurred during saving
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        return False

def load_session_state_from_dir(dashboard_path):
    """
    Loads the Streamlit session state from a published dashboard's directory.
    """
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

            st.success(f"Dashboard '{os.path.basename(dashboard_path)}' loaded successfully!")
        except Exception as e:
            st.error(f"Error loading dashboard: {e}")
    else:
        st.error("Dashboard configuration file not found.")

def delete_dashboard(dashboard_path):
    """Deletes a published dashboard's directory."""
    try:
        if os.path.exists(dashboard_path):
            shutil.rmtree(dashboard_path)
            st.success(f"Dashboard '{os.path.basename(dashboard_path)}' has been deleted successfully.")
        else:
            st.warning("Dashboard not found.")
    except Exception as e:
        st.error(f"Error deleting dashboard: {e}")

def switch_dashboard_privacy(dashboard_path):
    """Switches a dashboard between 'public' and 'private' privacy settings."""
    try:
        base_dir = "published_dashboards"
        current_privacy = "public" if "Public" in dashboard_path else "private"
        # The line below correctly handles the switch in both directions
        new_privacy = "private" if current_privacy == "public" else "public"
        dashboard_name = os.path.basename(dashboard_path)

        new_path = os.path.join(base_dir, new_privacy, dashboard_name)

        # Check if the destination path already exists
        if os.path.exists(new_path):
            st.warning(f"A dashboard named '{dashboard_name}' already exists in the '{new_privacy}' gallery. Cannot switch privacy.")
            return

        # Create the new privacy directory if it doesn't exist
        new_privacy_dir = os.path.dirname(new_path)
        if not os.path.exists(new_privacy_dir):
            os.makedirs(new_privacy_dir)

        shutil.move(dashboard_path, new_path)
        st.success(f"Dashboard '{dashboard_name}' has been moved to the '{new_privacy}' gallery.")

    except Exception as e:
        st.error(f"Error switching privacy: {e}")


def run_private_dashboard_gallery():
    """
    A Streamlit view to publish and load dashboards.
    """
    st.title("Publish Dashboard")
    st.markdown("Save your current dashboard configuration for future use or sharing.")

    dashboard_name = st.text_input("Enter a name for your dashboard:", help="This name will be used as the folder name. Avoid special characters.")

    privacy_setting = st.radio(
        "Who can see this dashboard?",
        ('Private', 'Public')
    )

    if st.button("Publish Dashboard"):
        if not dashboard_name:
            st.warning("Please enter a name for the dashboard.")
        else:
            with st.spinner("Publishing..."):
                # The 'save_session_state_to_dir' function is called here to save the dashboard.
                success = save_session_state_to_dir(dashboard_name, privacy_setting)
            if success:
                st.success(f"Dashboard '{dashboard_name}' has been published successfully as '{privacy_setting.lower()}'.")
                # st.balloons()
            else:
                st.error("Failed to publish the dashboard. Please check the console for details.")

    st.markdown("---")
    st.title("Manage Published Dashboards")
    st.markdown("Select a dashboard from the list below to load, delete, or change its privacy.")

    base_dir = "published_dashboards"

    # Reload dashboards after any action
    all_dashboards = []
    if os.path.exists(base_dir):
        for root, dirs, files in os.walk(base_dir):
            if "config.json" in files:
                rel_path = os.path.relpath(root, base_dir)
                privacy = rel_path.split(os.path.sep)[0]
                name = os.path.basename(root)
                full_path = root
                all_dashboards.append({"name": name, "path": full_path, "privacy": privacy})

    if all_dashboards:
        # Create a dictionary to map the display name to the full dashboard object
        display_to_dashboard_map = {f"{d['name']} ({d['privacy'].capitalize()})": d for d in all_dashboards}
        display_names = list(display_to_dashboard_map.keys())

        # Use a single radio button for all dashboards
        selected_display_name = st.radio(
            "Select a dashboard:",
            display_names,
            key="combined_dashboard_selector"
        )

        # Get the full dashboard object from the selected display name
        selected_dashboard = display_to_dashboard_map.get(selected_display_name)
        selected_path = selected_dashboard["path"]

        if selected_path:
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Load Dashboard"):
                    with st.spinner("Loading..."):
                        # This button triggers the 'load_session_state_from_dir' function.
                        load_session_state_from_dir(selected_path)
            with col2:
                # Determine the current privacy to set the button label
                current_privacy = selected_dashboard["privacy"]
                print(current_privacy)
                new_privacy = "private" if current_privacy == "Public" else "public"
                if st.button(f"Switch to {new_privacy.capitalize()}"):
                    with st.spinner("Switching..."):
                        # This button triggers the 'switch_dashboard_privacy' function.
                        switch_dashboard_privacy(selected_path)
                        st.rerun()
            with col3:
                delete_confirmed = st.checkbox("Confirm Deletion", key="delete_confirm")
                if st.button("Delete Dashboard", disabled=not delete_confirmed):
                    with st.spinner("Deleting..."):
                        # This button triggers the 'delete_dashboard' function.
                        delete_dashboard(selected_path)
        else:
            st.info("No dashboard selected.")
    else:
        st.info("No published dashboards found.")

# The user-facing view, which could be integrated into a Streamlit app
# as a page.
