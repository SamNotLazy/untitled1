


import json
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd

# You can keep these imports, as they might be necessary to deserialize
# objects from the loaded dashboard state, even though they are not used
# for the loading logic itself.
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from streamlit.errors import StreamlitValueAssignmentNotAllowedError
from xgboost import XGBRegressor

from pages.home import load_geo




# def save_session_state(CONFIG_FILE="config.json"):
#     """
#     Saves the current Streamlit session state to a JSON file.
#
#     This function handles specific non-serializable objects (like DataFrames,
#     GeoDataFrames, and machine learning models) by serializing them to files
#     on disk and saving the file paths.
#     """
#     try:
#         # Create a temporary directory for non-serializable objects
#         temp_dir = "Non-Serialisable Object Storage"
#         if not os.path.exists(temp_dir):
#             os.makedirs(temp_dir)
#
#         data_to_save = {}
#         for key, value in st.session_state.items():
#             # Special handling for non-serializable objects
#             if isinstance(value, gpd.GeoDataFrame):
#                 data_to_save[key] = {
#                     "value": st.session_state.get("selected_state"),
#                     "type": "GeoDataFrame_name"
#                 }
#             elif isinstance(value, pd.DataFrame):
#                 data_to_save[key] = {
#                     "value": value.to_dict('records'),
#                     "type": "DataFrame"
#                 }
#             # Handle non-serializable models and other objects with joblib
#             elif isinstance(value, (RandomForestRegressor, XGBRegressor, LinearRegression, StandardScaler, dict, pd.Series, np.ndarray)):
#                 # Save object to a joblib file
#                 file_path = os.path.join(temp_dir, f"{key}.joblib")
#                 joblib.dump(value, file_path)
#                 data_to_save[key] = {
#                     "value": file_path,
#                     "type": "joblib_file"
#                 }
#             else:
#                 try:
#                     # Attempt to serialize the value to check if it's JSON serializable
#                     json.dumps(value)
#                     # If successful, save the original value and its type
#                     data_to_save[key] = {
#                         "value": value,
#                         "type": type(value).__name__
#                     }
#                 except TypeError:
#                     # If not serializable, convert it to a string representation and save it
#                     data_to_save[key] = {
#                         "value": str(value),
#                         "type": "str"
#                     }
#
#         with open(CONFIG_FILE, "w") as f:
#             json.dump(data_to_save, f, indent=4)
#
#     except Exception as e:
#         st.error(f"Error saving session state: {e}")
#
# def load_session_state(CONFIG_FILE="config.json"):
#     """
#     Loads the Streamlit session state from a JSON file.
#
#     This function reads the JSON file and updates the session state. It
#     correctly handles specific non-serializable values by loading them from
#     their serialized files.
#     """
#     if os.path.exists(CONFIG_FILE):
#         try:
#             with open(CONFIG_FILE, "r") as f:
#                 loaded_data = json.load(f)
#
#             # Dictionary to map string type names back to their constructor functions
#             type_mapping = {
#                 'str': str, 'int': int, 'float': float, 'bool': bool,
#                 'list': list, 'dict': dict, 'tuple': tuple,
#             }
#
#             # Update current session state with loaded values
#             for key, item_data in loaded_data.items():
#                 if key in ["editor_linear", "multi_uploader",
#                            'FormSubmitter:agg_chooser_form-Apply aggregation & build grouped view',
#                            'FormSubmitter:bucket_add_form-Add Bucket', 'FormSubmitter:bucket_remove_form-Remove Selected',
#                            "combined_dashboard_selector"]:
#                     continue
#                 value = item_data.get("value")
#                 type_name = item_data.get("type")
#
#                 if type_name == "GeoDataFrame_name":
#                     # Load GeoDataFrame from the saved file path
#                     STATES_DIR = 'States'
#                     geo_path = f'{STATES_DIR}/{value}/{value}_DISTRICTS.geojson'
#                     gdf, GEO_COL = load_geo(geo_path)
#                     if gdf is not None:
#                         st.session_state["gdf"] = gdf
#                         st.session_state["GEO_COL"] = GEO_COL
#                         st.session_state["selected_state"] = value
#                     else:
#                         st.error(f"Failed to load GeoDataFrame for state '{value}'.")
#                 elif type_name == "DataFrame":
#                     # Convert the list of dictionaries back into a DataFrame
#                     st.session_state[key] = pd.DataFrame.from_dict(value)
#                 elif type_name == "joblib_file":
#                     # Load the serialized object from the file
#                     if os.path.exists(value):
#                         st.session_state[key] = joblib.load(value)
#                 else:
#                     # Get the constructor function for the saved type
#                     constructor = type_mapping.get(type_name)
#                     if constructor:
#                         # Cast the value to the correct type
#                         st.session_state[key] = constructor(value)
#                     else:
#                         st.session_state[key] = value
#             st.success("Session state successfully loaded!")
#         except json.JSONDecodeError:
#             st.error(f"Error loading session state: '{CONFIG_FILE}' is not a valid JSON file.")
#         except StreamlitValueAssignmentNotAllowedError:
#             pass
#         except Exception as e:
#             st.error(f"An unexpected error occurred while loading session state: {e}")
#     else:
#         st.warning(f"No saved session state found at '{CONFIG_FILE}'.")

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
                        # Assuming 'load_geo' is a function that returns the GeoDataFrame and its geometry column name
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

            if "loaded_dashboard_name" not in st.session_state:
                st.session_state["loaded_dashboard_name"] = os.path.basename(dashboard_path)
            else:
                st.session_state["loaded_dashboard_name"] = os.path.basename(dashboard_path)
            st.success(f"Dashboard '{os.path.basename(dashboard_path)}' loaded successfully!")


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
    # save_session_state()
    st.title("Load Public Dashboards")
    st.markdown("Select a publicly saved dashboard from the list below to load its configuration.")

    #public_dir = "published_dashboards/public"
    #public_dir = "published_dashboards/Public"
    base = "published_dashboards"
    cands = [os.path.join(base, "public"), os.path.join(base, "Public")]
    public_dir = next((p for p in cands if os.path.isdir(p)), cands[0])  # pick whichever exists; default to lower
    os.makedirs(public_dir, exist_ok=True)  # create if missing

    if os.path.exists(public_dir):
        # List all directories within the public dashboard folder
        dashboard_names = [d for d in os.listdir(public_dir) if os.path.isdir(os.path.join(public_dir, d))]
        dashboard_names.sort()

        if dashboard_names:
            # Function to handle button clicks
            def handle_load_click(dashboard_path):
                with st.spinner("Loading..."):
                    load_session_state_from_dir(dashboard_path)
                # st.rerun()

            # Create a button for each dashboard
            for name in dashboard_names:
                selected_path = os.path.join(public_dir, name)

                # Use columns to place the name and button on the same line
                col1, col2 = st.columns([0.7, 0.3])

                with col1:
                    st.markdown(f"**{name}**")

                with col2:
                    if st.button("Load", key=f"load_button_{name}"):
                        handle_load_click(selected_path)
                        st.session_state.logged_in = False

        else:
            st.info("No public dashboards found.")
    else:
        st.info("The public dashboards directory does not exist. Please ensure a 'public' folder exists inside 'published_dashboards' with saved dashboards.")


# This block ensures the script can be run directly as a Streamlit app
if __name__ == "__main__":
    run_public_dashboard_view()
