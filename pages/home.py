import io
import json
import keyword
import os
import re

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from streamlit.errors import StreamlitValueAssignmentNotAllowedError
from xgboost import XGBRegressor

from utils.data_prep import fuzzy_map, clean_df


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
#                     "value": st.session_state["selected_state"],
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
#                 st.info(f"Successfully serialized '{key}' to {file_path}")
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
#                     st.warning(f"Non-serializable object found for key '{key}'. Saving as a string representation.")
#                     data_to_save[key] = {
#                         "value": str(value),
#                         "type": "str"
#                     }
#
#         with open(CONFIG_FILE, "w") as f:
#             json.dump(data_to_save, f, indent=4)
#
#         st.success(f"Session state successfully saved to '{CONFIG_FILE}'!")
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
#                 if key in ["editor_linear","multi_uploader"
#                     ,'FormSubmitter:agg_chooser_form-Apply aggregation & build grouped view',
#                            'FormSubmitter:bucket_add_form-Add Bucket','FormSubmitter:bucket_remove_form-Remove Selected' ]:
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
#                         st.info(f"Successfully loaded '{key}' from {value}")
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

# ---------- Load GeoJSON once ----------
@st.cache_data
def load_geo(file_path: str):
    g = gpd.read_file(file_path)
    col = "dtname"                       # adjust if your join column has a different name
    g[col] = g[col].astype(str).str.strip().str.upper()
    return g, col


# ---------- Helpers for multi-file fuzzy merge ----------
def _read_any(file):
    """Read csv/xlsx/xls to DataFrame with gentle cleaning."""
    try:
        if file.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)
    except Exception as e:
        raise RuntimeError(f"Could not read {file.name}: {e}")
    # drop fully-empty rows early
    df = df.dropna(how="all")
    return clean_df(df)


def _normalize_key_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Normalize a join key for reliable PK checks & matching."""
    s = df[col]
    s = s.astype(str).str.strip().str.upper()
    # treat visually empty / stringified NaN as missing
    s = s.replace({"": pd.NA, "NAN": pd.NA, "nan": pd.NA})
    return s


def _check_primary_key(df: pd.DataFrame, col: str) -> bool:
    """True if normalized key has no NA and is unique."""
    s = _normalize_key_series(df, col)
    return s.notna().all() and s.is_unique


def _key_stats(df: pd.DataFrame, col: str):
    """Return (#NA, #duplicated_rows, list_of_duplicated_values) on normalized key."""
    s = _normalize_key_series(df, col)
    na = int(s.isna().sum())
    dup_rows = int(s.duplicated().sum())
    dup_vals = s[s.duplicated(keep=False)].dropna().unique().tolist()
    return na, dup_rows, dup_vals


def _fuzzy_key_map(left_series: pd.Series, right_series: pd.Series, thresh: int):
    """
    Map LEFT values to closest RIGHT values using fuzzy_map.
    Drop nulls BEFORE casting to str so we don't create 'NAN' strings,
    and scrub any leftover 'NAN' rows from the preview table.
    """
    left_norm  = pd.Series(left_series).dropna().astype(str).str.strip().str.upper()
    right_norm = pd.Series(right_series).dropna().astype(str).str.strip().str.upper()

    left_vals  = left_norm.unique().tolist()
    right_vals = right_norm.unique().tolist()

    mapping, unmatched_left, match_df = fuzzy_map(left_vals, right_vals, thresh=thresh)

    matched_right   = set(v for v in mapping.values() if pd.notna(v))
    right_unmatched = sorted(list(set(right_vals) - matched_right))

    # Clean preview (defensive): remove bogus NAN rows if producer created them
    if {"Uploaded Name", "Matched Name"}.issubset(match_df.columns):
        match_df = (
            match_df.replace({"NAN": pd.NA, "nan": pd.NA})
            .dropna(subset=["Uploaded Name", "Matched Name"])
        )

    return mapping, unmatched_left, match_df, right_unmatched


def _apply_mapping_for_merge(left_df: pd.DataFrame, right_df: pd.DataFrame,
                             left_on: str, right_on: str, join: str, thresh: int):
    """
    Return (merged_df, diagnostics) using fuzzy key alignment.
    We normalize keys, map LEFT keys into RIGHT domain, then merge on the mapped key.
    """
    # normalize with the same function used by PK checks
    lkey = _normalize_key_series(left_df, left_on)
    rkey = _normalize_key_series(right_df, right_on)

    mapping, left_unmatched, match_df, right_unmatched = _fuzzy_key_map(lkey, rkey, thresh)

    L = left_df.copy()
    R = right_df.copy()
    L["__key_src_left"]  = left_df[left_on]
    R["__key_src_right"] = right_df[right_on]
    L["__merge_key"] = lkey.map(mapping)
    R["__merge_key"] = rkey

    merged = pd.merge(L, R, on="__merge_key", how=join, suffixes=("_L", "_R"))
    diags = {
        "match_df": match_df,
        "left_unmatched": left_unmatched,
        "right_unmatched": right_unmatched,
        "mapping_size": len(mapping),
    }
    return merged, diags


def _slug(name: str) -> str:
    # Lowercase; non-alnum → underscore; trim extra underscores
    s = re.sub(r'[^0-9a-zA-Z_]+', '_', str(name).strip()).strip('_')
    # Collapse multiple underscores
    s = re.sub(r'_+', '_', s)
    return s.lower()

def _auto_backtick_quoted(expr: str, columns_set):
    """
    Convert 'Col' or "Col" to `Col` ONLY if it exactly matches a column.
    """
    def repl(m):
        txt = m.group(2)
        return f"`{txt}`" if txt in columns_set else m.group(0)
    return re.sub(r"(['\"])(.*?)\1", repl, expr)

def _compile_formula(expr: str, df):
    """
    Turn a user formula into a safe python expression.
    Supports:
      - bare identifiers (col1*col2)
      - `backticked names`
      - 'quoted' or "quoted" if they match a column (auto-converted)
      - slug aliases: every column also available as a python-safe variable
    """
    cols = {c: df[c] for c in df.columns}
    col_names = set(cols.keys())

    # 1) Auto-convert matching 'quoted'/"quoted" → backticks
    expr = _auto_backtick_quoted(expr, col_names)

    # 2) Expose aliases like my_column → df['My Column']
    aliases = { _slug(c): c for c in col_names }
    # Don’t shadow reserved words / python keywords
    aliases = { a:c for a,c in aliases.items() if a and not keyword.iskeyword(a) and a not in {"np", "cols"} }

    # 3) Replace *exact* bare identifiers that are exact column names (no spaces) → cols['name']
    #    Leave alias names as-is (they’ll exist in locals).
    def replace_token(m):
        tok = m.group(0)
        if tok in col_names:
            return f"cols[{tok!r}]"
        # if token is a known function/const, keep
        if tok in {"np"} or keyword.iskeyword(tok):
            return tok
        # else leave as-is (it could be an alias or variable)
        return tok

    expr = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", replace_token, expr)

    # 4) Finally convert remaining backticked names → cols['name']
    expr = re.sub(r"`([^`]+)`", lambda m: f"cols[{m.group(1)!r}]", expr)

    # Locals: give access to cols + alias variables (each alias is a Series)
    local_vars = {"cols": cols, **{ a: cols[c] for a,c in aliases.items() }}

    # Globals: NumPy only
    global_vars = {"__builtins__": {}, "np": np}
    return expr, global_vars, local_vars



def run_home():
    # --- State Selection ---
    # --- ADDED BUTTONS FOR SAVE/LOAD ---
    # col1, col2 = st.columns(2)
    # with col1:
    #     if st.button("💾 Save Session"):
    #         save_session_state()
    # with col2:
    #     if st.button("📂 Load Session"):
    #         load_session_state()
    # # --- END ADDED BUTTONS ---

    STATES_DIR = 'States'
    try:
        state_list = [name for name in os.listdir(STATES_DIR) if os.path.isdir(os.path.join(STATES_DIR, name))]
        state_list.sort()
    except FileNotFoundError:
        st.error(f"Directory not found: '{STATES_DIR}'")
        state_list = []

    if not state_list:
        st.stop()

    st.title("Data Preparation")

    # Persist selected_state across pages/reruns
    if "selected_state" not in st.session_state and state_list:
        st.session_state["selected_state"] = state_list[0]

    selected_state = st.selectbox(
        "**Select a State**",
        state_list,
        index=state_list.index(st.session_state["selected_state"])
        if st.session_state["selected_state"] in state_list else 0,
    )
    st.session_state["selected_state"]=selected_state
    # selected_state = st.session_state["selected_state"]

    # --- Load GeoData ---
    geo_path = f'{STATES_DIR}/{selected_state}/{selected_state}_DISTRICTS.geojson'
    gdf, GEO_COL = load_geo(geo_path)
    st.session_state["gdf"] = gdf
    st.session_state["GEO_COL"] = GEO_COL
    valid_geo_unit = sorted(gdf[GEO_COL].unique())
    geo_unit = "District"

    # ---------------- Multi-file Upload & Merge Workflow ----------------
    st.sidebar.header("📂 Data files")
    files = st.sidebar.file_uploader(
        f"Upload one or more {geo_unit}-level data files (CSV/Excel)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key="multi_uploader",
    )

    if "ingested_ids" not in st.session_state:
        st.session_state.ingested_ids = set()

    def _fid(f):
        return f"{getattr(f, 'name', '')}:{getattr(f, 'size', 0)}"

    if "merge_pool" not in st.session_state:
        st.session_state.merge_pool = {}     # name -> df
    if "merge_history" not in st.session_state:
        st.session_state.merge_history = []  # record each merge step

    if files:
        for f in files:
            fid = _fid(f)
            if fid in st.session_state.ingested_ids:
                continue
            st.session_state.ingested_ids.add(fid)

            name = os.path.splitext(os.path.basename(f.name))[0]
            if name in st.session_state.merge_pool:
                i, base = 2, name
                while f"{base}_{i}" in st.session_state.merge_pool:
                    i += 1
                name = f"{base}_{i}"
            try:
                st.session_state.merge_pool[name] = _read_any(f)
            except RuntimeError as e:
                st.sidebar.error(str(e))

    # Show pool only after at least one file is ingested
    if not st.session_state.merge_pool:
        return

    st.markdown("### 📚 Current tables in pool")
    with st.expander("Show table previews"):
        for nm, df in st.session_state.merge_pool.items():
            st.caption(f"**{nm}** — shape {df.shape}")
            st.dataframe(df.head(10), use_container_width=True)

    pool_names_all = sorted(st.session_state.merge_pool.keys())
    pool_len = len(pool_names_all)

    # --- Force recursive merging until ONE table remains ----------------
    if pool_len > 1:
        remaining_msg = (
            f"Please also merge the table **{pool_names_all[0]}**"
            if pool_len == 2 else
            "Please continue merging until only one table remains."
        )
        st.warning(f"🧩 {remaining_msg}  \nCurrent tables: {', '.join(pool_names_all)}")

        # If previous run wrote *_next_* values, move them into widget keys now.
        if "_next_left" in st.session_state:
            st.session_state["merge_left_sel"] = st.session_state.pop("_next_left")
        if "_next_right" in st.session_state:
            st.session_state["merge_right_sel"] = st.session_state.pop("_next_right")

        st.markdown("### 🔗 Merge two tables (fuzzy)")
        c1, c2 = st.columns(2)

        default_left  = st.session_state.get("merge_left_sel", pool_names_all[0])
        if default_left not in pool_names_all:
            default_left = pool_names_all[0]

        with c1:
            left_name = st.selectbox(
                "Left table",
                pool_names_all,
                index=pool_names_all.index(default_left),
                key="merge_left_sel"
            )

        right_candidates = [n for n in pool_names_all if n != left_name]
        default_right = st.session_state.get("merge_right_sel", right_candidates[0])
        if default_right not in right_candidates:
            default_right = right_candidates[0]

        with c2:
            right_name = st.selectbox(
                "Right table",
                right_candidates,
                index=right_candidates.index(default_right),
                key="merge_right_sel"
            )

        left_df  = st.session_state.merge_pool[left_name]
        right_df = st.session_state.merge_pool[right_name]

        left_on  = st.selectbox("Left key column",  left_df.columns,  key="merge_left_on")
        right_on = st.selectbox("Right key column", right_df.columns, key="merge_right_on")

        # --- Per-merge similarity slider (default 80%) ---
        sim_key = f"sim_thresh::{left_name}::{right_name}"
        sim_thresh = st.slider(
            "🔎 Similarity threshold for this merge",
            50, 100, 80, 1, key=sim_key,
            help="Used for the fuzzy alignment between the selected tables."
        )

        _left_norm = _normalize_key_series(left_df, left_on)
        _right_norm = _normalize_key_series(right_df, right_on)
        left_df  = left_df.loc[_left_norm.notna()].copy()
        right_df = right_df.loc[_right_norm.notna()].copy()

        left_pk  = _check_primary_key(left_df, left_on)
        right_pk = _check_primary_key(right_df, right_on)

        l_na, l_dup_rows, l_dup_vals = _key_stats(left_df, left_on)
        r_na, r_dup_rows, r_dup_vals = _key_stats(right_df, right_on)

        if left_pk or right_pk:
            st.success(f"Primary key check ✓ — {'left' if left_pk else 'right'} side is unique and non-null (after normalization).")
        else:
            st.error("At least one key column is NOT a primary key (unique, no missing) after normalization. "
                     "Clean the keys or override to allow many-to-many merges.")

        st.caption(
            f"Key diagnostics (normalized) → "
            f"Left: NA={l_na}, duplicated rows={l_dup_rows} | "
            f"Right: NA={r_na}, duplicated rows={r_dup_rows}"
        )
        if l_dup_vals:
            st.caption(f"Left duplicated values (sample): {', '.join(map(str, l_dup_vals[:10]))}" + (" …" if len(l_dup_vals) > 10 else ""))
        if r_dup_vals:
            st.caption(f"Right duplicated values (sample): {', '.join(map(str, r_dup_vals[:10]))}" + (" …" if len(r_dup_vals) > 10 else ""))

        allow_nonpk = st.checkbox(
            "Override primary-key requirement (allow many-to-many merge)",
            value=False,
            help="Enable this if you intend a many-to-many join (row multiplication possible).",
        )

        join_type = st.selectbox("Join type", ["inner", "left", "right", "outer"], index=1)

        # Diagnostics & editable correction UIs
        st.markdown("#### 🧭 Match diagnostics & corrections")
        with st.spinner("Running fuzzy mapping preview..."):
            _preview, diags_prev = _apply_mapping_for_merge(
                left_df, right_df, left_on, right_on, join_type, sim_thresh
            )

        st.caption("**Top of similarity table** (preview)")
        st.dataframe(diags_prev["match_df"], use_container_width=True)

        no_unmatched = not diags_prev["left_unmatched"] and not diags_prev["right_unmatched"]

        if no_unmatched:
            st.info("All keys matched at the current threshold. No corrections needed.")
            left_ed = right_ed = pd.DataFrame()
            button_label = "Merge tables"
        else:
            if diags_prev["left_unmatched"]:
                left_un_df = pd.DataFrame({
                    "left_unmatched": diags_prev["left_unmatched"],
                    "new_value":      diags_prev["left_unmatched"],
                })
                st.markdown("**Left unmatched values** — edit `new_value` to fix")
                left_ed = st.data_editor(left_un_df, use_container_width=True, num_rows="dynamic", key="left_editor")
            else:
                left_ed = pd.DataFrame(columns=["left_unmatched", "new_value"])

            if diags_prev["right_unmatched"]:
                right_un_df = pd.DataFrame({
                    "right_unmatched": diags_prev["right_unmatched"],
                    "new_value":       diags_prev["right_unmatched"],
                })
                st.markdown("**Right unmatched values** — edit `new_value` to fix")
                right_ed = st.data_editor(right_un_df, use_container_width=True, num_rows="dynamic", key="right_editor")
            else:
                right_ed = pd.DataFrame(columns=["right_unmatched", "new_value"])

            button_label = "Apply corrections & perform merge"

        # ---------- APPLY CORRECTIONS & MERGE ----------
        can_merge = (left_pk or right_pk) or allow_nonpk or no_unmatched

        if st.button(button_label, type="primary", disabled=not can_merge):
            try:
                # Defensive copies
                L, R = left_df.copy(), right_df.copy()

                # Apply user corrections safely
                if not no_unmatched:
                    if not left_ed.empty and "left_unmatched" in left_ed and "new_value" in left_ed:
                        for _, row in left_ed.iterrows():
                            old = str(row.get("left_unmatched", "") or "").strip()
                            new = str(row.get("new_value", "") or "").strip()
                            if old and new and old != new:
                                esc_old = pd.Series([old]).str.replace(r'([\\.^$|?*+()\[\]{}])', r'\\\1', regex=True)[0]
                                L[left_on] = L[left_on].astype(str).str.replace(rf"^\s*{esc_old}\s*$", new, regex=True)

                    if not right_ed.empty and "right_unmatched" in right_ed and "new_value" in right_ed:
                        for _, row in right_ed.iterrows():
                            old = str(row.get("right_unmatched", "") or "").strip()
                            new = str(row.get("new_value", "") or "").strip()
                            if old and new and old != new:
                                esc_old = pd.Series([old]).str.replace(r'([\\.^$|?*+()\[\]{}])', r'\\\1', regex=True)[0]
                                R[right_on] = R[right_on].astype(str).str.replace(rf"^\s*{esc_old}\s*$", new, regex=True)

                # Perform merge using the *per-merge* threshold
                merged_df, diags = _apply_mapping_for_merge(
                    L, R, left_on, right_on, join_type, sim_thresh
                )

                # Update pool/state
                step_name = f"Merged[{left_name}+{right_name}]"
                st.session_state.merge_history.append({
                    "left": left_name, "right": right_name,
                    "left_on": left_on, "right_on": right_on,
                    "join": join_type, "threshold": sim_thresh,
                    "result": step_name, "diags": diags,
                })

                st.session_state.merge_pool.pop(left_name, None)
                st.session_state.merge_pool.pop(right_name, None)
                st.session_state.merge_pool[step_name] = merged_df

                # Prefill next selections
                remaining = sorted([n for n in st.session_state.merge_pool.keys() if n != step_name])
                st.session_state["_next_left"] = step_name
                st.session_state["_next_right"] = (remaining[0] if remaining else None)

                st.success(f"Created `{step_name}` — pool now has {len(st.session_state.merge_pool)} tables.")
                st.rerun()

            except Exception as e:
                st.error("Merge failed — see details below.")
                st.exception(e)

        st.stop()  # never fall through to column selection while pool_len > 1

    # ---------------------- Exactly ONE table remains ----------------------
    st.markdown("### ✅ Working dataset ready for column selection & mapping")
    only_name = next(iter(st.session_state.merge_pool.keys()))
    working_df = st.session_state.merge_pool[only_name].copy()
    st.session_state["uploaded_df"] = working_df.copy()

    # (Column-selection & mapping) -----------------------------------------
    required_keys = ["geo_unit_col", "target_col", "factor_cols", "uploaded_df"]
    if not st.session_state.get("columns_confirmed", False) or any(k not in st.session_state for k in required_keys):
        st.session_state.columns_confirmed = False

    if not st.session_state.columns_confirmed:
        st.markdown("### 🔧 Column Selection")

        # 1) Pick District column
        geo_unit_col = st.selectbox(
            f"1️⃣ Select the **{geo_unit} column**",
            working_df.columns,
            index=0
        )
        working_df = working_df.dropna(subset=[geo_unit_col])

        # Prepare normalized series for preview
        geo_series_norm = working_df[geo_unit_col].astype(str).str.strip().str.upper()

        # 2) Shapefile threshold slider (moved here)
        geo_thresh = st.slider(
            f"🔎 Shapefile matching threshold (for {geo_unit} names)",
            50, 100, st.session_state.get("geo_sim_thresh", 80), 1,
            help="Used to match your uploaded names to the shapefile names."
        )

        # 3) Live similarity preview + editable unmatched
        preview_map, preview_unmatched, preview_match_df = fuzzy_map(
            geo_series_norm.unique(), valid_geo_unit, thresh=geo_thresh
        )
        st.markdown(f"#### {geo_unit}-Name Similarity Preview")
        st.dataframe(preview_match_df, use_container_width=True)

        # Editable corrections table for unmatched uploaded names
        if preview_unmatched:
            st.markdown("**Unmatched uploaded names** — edit `new_value` to fix")
            geo_left_un_df = pd.DataFrame({
                "uploaded_unmatched": sorted(preview_unmatched),
                "new_value":          sorted(preview_unmatched),
            })
            geo_left_ed = st.data_editor(
                geo_left_un_df,
                use_container_width=True,
                num_rows="dynamic",
                key="geo_left_editor"
            )
        else:
            geo_left_ed = pd.DataFrame(columns=["uploaded_unmatched", "new_value"])

        # “Missing in upload” info (right side)
        present_geo_preview = set(pd.Series(list(preview_map.values())).dropna().unique())
        missing_in_geo_preview = sorted(set(valid_geo_unit) - present_geo_preview)
        if missing_in_geo_preview:
            st.caption(f"**Geo {geo_unit}s missing in upload**")
            st.code(", ".join(missing_in_geo_preview))


        # 4) Target + Factors (with “select all numeric” helper + target direction)
        target_col = st.selectbox(
            "2️⃣ Select the **Target indicator**",
            [col for col in working_df.columns if col != geo_unit_col],
            index=0
        )

        # Helper list of numeric candidate factors (exclude geo + target)
        numeric_cols = working_df.select_dtypes(include="number").columns.tolist()
        candidate_numeric = [c for c in numeric_cols if c not in [geo_unit_col, target_col]]

        use_all_numeric = st.checkbox(
            "Select all numeric columns as factors",
            value=False,
            help="Quickly include every numeric column (except the geo and target)."
        )

        available_factors = [col for col in working_df.columns if col not in [geo_unit_col, target_col]]
        default_factors = candidate_numeric if use_all_numeric else []

        with st.form("column_selection_form"):
            # Target direction (restored)
            target_direction = st.radio(
                "🎯 What is the **desired direction** for the Target Indicator?",
                options=["Increase", "Decrease", "No Preference"],
                horizontal=True,
                index=(
                    0 if "target_direction" not in st.session_state
                    else ["Increase", "Decrease", "No Preference"].index(st.session_state["target_direction"])
                ),
                help=(
                    "Increase: higher target values are desirable (e.g., literacy rate).  "
                    "Decrease: lower target values are desirable (e.g., poverty rate).  "
                    "No Preference: neutral."
                )
            )

            factor_cols = st.multiselect(
                "3️⃣ Select **Factor columns** (independent variables)",
                options=available_factors,
                default=default_factors
            )

            if st.form_submit_button("Confirm Selections"):
                # Apply user corrections to the geo column BEFORE saving selections
                if not geo_left_ed.empty and "uploaded_unmatched" in geo_left_ed and "new_value" in geo_left_ed:
                    col_upper = working_df[geo_unit_col].astype(str).str.strip().str.upper()
                    for _, row in geo_left_ed.iterrows():
                        old = str(row.get("uploaded_unmatched", "") or "").strip().upper()
                        new = str(row.get("new_value", "") or "").strip()
                        if old and new and old != new:
                            mask = col_upper == old
                            working_df.loc[mask, geo_unit_col] = new
                            # refresh normalized view after edits
                            col_upper = working_df[geo_unit_col].astype(str).str.strip().str.upper()

                # Normalize geo col and persist selections
                working_df[geo_unit_col] = working_df[geo_unit_col].astype(str).str.strip().str.upper()

                st.session_state["geo_unit_col"] = geo_unit_col
                st.session_state["target_col"] = target_col
                st.session_state["factor_cols"] = factor_cols
                st.session_state["uploaded_df"] = clean_df(working_df)

                # ✅ persist the chosen target direction
                st.session_state["target_direction"] = target_direction

                # persist the chosen shapefile threshold too (you already set the slider above)
                st.session_state["geo_sim_thresh"] = geo_thresh

                st.session_state.columns_confirmed = True
                st.rerun()


        st.stop()
    else:
        st.markdown("### ✅ Current Column Selections")
        st.write(f"• **{geo_unit} Column**: `{st.session_state['geo_unit_col']}`")
        st.write(f"• **Target Indicator**: `{st.session_state['target_col']}`")
        st.write(f"• **Factor Columns**: {', '.join(st.session_state['factor_cols'])}")

        if st.button("✏️ Edit Selections"):
            st.session_state.columns_confirmed = False
            st.rerun()

    # ---------- After Confirmation ----------
    df_up = st.session_state["uploaded_df"]
    geo_unit_col = st.session_state["geo_unit_col"]
    target_col = st.session_state["target_col"]
    factor_cols = st.session_state["factor_cols"]

    df_up[geo_unit_col] = df_up[geo_unit_col].astype(str).str.strip().str.upper()

    # Final mapping with the stored threshold (includes any corrections you made)
    mapping, unmatched_upload, match_df = fuzzy_map(
        df_up[geo_unit_col].unique(),
        valid_geo_unit,
        thresh=st.session_state.get("geo_sim_thresh", 80)
    )
    df_up_mapped = df_up.copy()
    df_up_mapped[geo_unit_col] = df_up_mapped[geo_unit_col].map(mapping)
    present_geo = set(df_up_mapped[geo_unit_col].dropna().unique())
    missing_in_geo = sorted(set(valid_geo_unit) - present_geo)

    st.session_state["mapping"] = mapping
    st.session_state["match_df"] = match_df
    st.session_state["unmatched_upload"] = unmatched_upload
    st.session_state["missing_geo"] = missing_in_geo

    st.warning(
        f"⚠️ The app performs **similarity-based matching** (threshold = {st.session_state.get('geo_sim_thresh', 80)}%). "
        f"{geo_unit}s below the threshold will appear blank on the map."
    )

    with st.expander(f"📍 Expected {geo_unit} names (from shapefile)"):
        st.markdown(
            f"""
            <div style="max-height: 200px; overflow-y: auto; background-color: #f0f0f0; padding: 0.5rem; border-radius: 5px;">
            {", ".join(valid_geo_unit)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(f"### {geo_unit}-Name Similarity Table")
    st.dataframe(match_df, use_container_width=True)

    if unmatched_upload:
        st.markdown("#### Uploaded names **not matched** (blank on map)")
        st.code(", ".join(unmatched_upload))
    if missing_in_geo:
        st.markdown(f"#### Geo {geo_unit}s **missing in upload**")
        st.code(", ".join(missing_in_geo))




    # ---------- Calculated Fields (Feature Engineering) ----------
    st.markdown("### 🧮 Calculated Fields")
    # Guard
    if "uploaded_df" not in st.session_state or st.session_state["uploaded_df"] is None:
        st.info("Upload data to create calculated fields.")
    else:
        # Registry of fields
        if "engineered_fields" not in st.session_state:
            st.session_state.engineered_fields = []  # list of {"name":..., "expr":...}

        # Re-apply previously created fields in order (idempotent over the latest df)
        base_df = st.session_state["uploaded_df"].copy()
        for fld in st.session_state.engineered_fields:
            try:
                expr_py, gvars, lvars = _compile_formula(fld["expr"], base_df)
                val = eval(expr_py, gvars, lvars)
                base_df[fld["name"]] = val
            except Exception as e:
                st.warning(f"Could not re-apply calculated field `{fld['name']}` ({fld['expr']}): {e}")

        # ---- Auto-add engineered numeric columns to factor list (if columns are confirmed) ----
        if st.session_state.get("columns_confirmed", False):
            geo_col = st.session_state.get("geo_unit_col")
            tgt_col = st.session_state.get("target_col")
            engineered_names = [f["name"] for f in st.session_state.engineered_fields if f["name"] in base_df.columns]
            engineered_numeric = [
                c for c in engineered_names
                if c not in (geo_col, tgt_col) and pd.api.types.is_numeric_dtype(base_df[c])
            ]
            cur_factors = list(st.session_state.get("factor_cols", []))
            for c in engineered_numeric:
                if c not in cur_factors:
                    cur_factors.append(c)
            st.session_state["factor_cols"] = cur_factors

        # Push re-applied df back so downstream sees the engineered columns
        st.session_state["uploaded_df"] = base_df

        with st.expander("➕ Create a new calculated field", expanded=False):
            st.markdown(
                "You can reference columns as **bare names** (`col1*col2`), with **backticks** "
                "(e.g., `` `Col A` - `Col B` ``), or even with quotes (auto-converted if they match). "
                "NumPy is available as `np`.\n\n"
                "**Examples**:\n"
                "- `col1*col2`\n"
                "- ``(`Col A` - `Col B`) / `Col B` * 100``\n"
                "- `np.log1p(col1)`\n"
                "- ``np.where(`Flag` > 0, 1, 0)``"
            )

            # Live helpers
            st.caption("Available columns:")
            st.code(", ".join(st.session_state["uploaded_df"].columns))

            st.caption("Aliases (python-safe names you can use directly):")
            alias_preview = ", ".join(sorted({_slug(c) for c in st.session_state["uploaded_df"].columns}))
            st.code(alias_preview or "(none)")

            new_name = st.text_input("New column name", placeholder="e.g., growth_pct")
            new_expr = st.text_input(
                "Formula",
                placeholder="e.g., (this_year - last_year) / last_year * 100"
            )

            if st.button("Add calculated field", type="primary", use_container_width=False):
                name = (new_name or "").strip()
                expr = (new_expr or "").strip()
                if not name:
                    st.error("Please provide a column name.")
                elif not expr:
                    st.error("Please provide a formula.")
                elif name in st.session_state["uploaded_df"].columns:
                    st.error(f"A column named `{name}` already exists.")
                else:
                    try:
                        temp = st.session_state["uploaded_df"].copy()
                        expr_py, gvars, lvars = _compile_formula(expr, temp)
                        val = eval(expr_py, gvars, lvars)

                        # Series or scalar broadcast
                        if hasattr(val, "__len__") and len(val) == len(temp):
                            temp[name] = val
                        else:
                            temp[name] = val  # pandas will broadcast scalars

                        # Persist
                        st.session_state["uploaded_df"] = temp
                        st.session_state.engineered_fields.append({"name": name, "expr": expr})

                        # If columns already confirmed, auto-append numeric engineered field to factor_cols
                        if st.session_state.get("columns_confirmed", False):
                            geo_col = st.session_state.get("geo_unit_col")
                            tgt_col = st.session_state.get("target_col")
                            if name not in (geo_col, tgt_col) and pd.api.types.is_numeric_dtype(temp[name]):
                                cur_factors = list(st.session_state.get("factor_cols", []))
                                if name not in cur_factors:
                                    cur_factors.append(name)
                                    st.session_state["factor_cols"] = cur_factors

                        # Force regroup later so the new column appears in grouped view
                        st.session_state.pop("grouped_df", None)

                        st.success(f"Added calculated field `{name}`.")
                        st.rerun()

                    except Exception as e:
                        st.exception(e)

        # Manage existing calculated fields
        if st.session_state.engineered_fields:
            st.markdown("#### 🗂️ Existing calculated fields")
            for fld in st.session_state.engineered_fields:
                st.write(f"- **{fld['name']}** = `{fld['expr']}`")

            col_del1, col_del2 = st.columns([3, 1])
            with col_del1:
                to_delete = st.selectbox(
                    "Remove a calculated field",
                    ["(Choose)"] + [f["name"] for f in st.session_state.engineered_fields],
                    index=0,
                    key="calc_del_sel"
                )
            with col_del2:
                if st.button("Delete", disabled=(to_delete == "(Choose)")):
                    name = to_delete
                    # Remove from registry
                    st.session_state.engineered_fields = [
                        f for f in st.session_state.engineered_fields if f["name"] != name
                    ]
                    # Drop the column from the dataframe if present
                    if name in st.session_state["uploaded_df"].columns:
                        st.session_state["uploaded_df"] = st.session_state["uploaded_df"].drop(columns=[name])
                    # Also remove from factor_cols if present
                    if "factor_cols" in st.session_state:
                        st.session_state["factor_cols"] = [f for f in st.session_state["factor_cols"] if f != name]
                    # Invalidate grouped view
                    st.session_state.pop("grouped_df", None)

                    st.success(f"Removed calculated field `{name}`.")
                    st.rerun()




    # ---------- Statistical Summary (ALL factor columns vs target) ----------
    st.markdown("### 📑 Statistical Summary (ALL features vs target)")

    # Pull context
    df_final = st.session_state["uploaded_df"].copy()
    geo_unit_col = st.session_state["geo_unit_col"]
    target_col = st.session_state["target_col"]

    # Determine the set of "all features":
    fallback_all_feats = [c for c in df_final.columns if c not in [geo_unit_col, target_col]]
    all_feature_cols = st.session_state.get("factor_cols", fallback_all_feats)

    # Keep only needed columns, coerce to numeric for stats
    work_cols = [c for c in all_feature_cols if c in df_final.columns] + [target_col]
    if target_col not in df_final.columns:
        st.info("Statistical summary unavailable — target column missing.")
    else:
        # Try SciPy for Pearson; fallback to statsmodels if unavailable
        try:
            from scipy.stats import pearsonr
            _HAS_SCIPY = True
        except Exception:
            _HAS_SCIPY = False

        num_df = df_final[work_cols].apply(pd.to_numeric, errors="coerce").dropna(subset=[target_col])
        if num_df.empty:
            st.warning("Not enough valid numeric data to compute statistics.")
        else:
            rows = []
            for col in all_feature_cols:
                if col not in num_df.columns:
                    rows.append({"Column Name": col, "p-value": float("nan"),
                                 "p-value<0.05": False, "r(Pearson Corr)": float("nan"), "R2": float("nan")})
                    continue

                x = num_df[col]
                y = num_df[target_col]
                valid = x.notna() & y.notna()
                xv, yv = x[valid], y[valid]

                if len(xv) >= 3 and xv.nunique() > 1 and yv.nunique() > 1:
                    if _HAS_SCIPY:
                        r, p = pearsonr(xv, yv)
                    else:
                        X_ = sm.add_constant(xv)
                        model_ = sm.OLS(yv, X_).fit()
                        r = xv.corr(yv)
                        p = model_.pvalues.get(col, float("nan"))
                    r2 = r**2 if pd.notna(r) else float("nan")
                else:
                    r, p, r2 = float("nan"), float("nan"), float("nan")

                rows.append({
                    "Column Name": col,
                    "p-value": p,
                    "p-value<0.05": (pd.notna(p) and p < 0.05),
                    "r(Pearson Corr)": r,
                    "R2": r2
                })

            summary_all_df = pd.DataFrame(rows)

            # Tidy formatting
            def _round(series, nd=9):
                return pd.to_numeric(series, errors="coerce").round(nd)

            summary_all_df["p-value"] = _round(summary_all_df["p-value"])
            summary_all_df["r(Pearson Corr)"] = _round(summary_all_df["r(Pearson Corr)"])
            summary_all_df["R2"] = _round(summary_all_df["R2"])

            st.dataframe(summary_all_df, use_container_width=True)

            # Download
            with st.expander("📤 Download Statistical Summary (ALL features)"):
                fmt = st.radio("Choose format:", ["CSV", "Excel"], horizontal=True, key="stat_summary_all_fmt")
                if fmt == "CSV":
                    st.download_button(
                        "⬇️ Download CSV",
                        data=summary_all_df.to_csv(index=False).encode("utf-8"),
                        file_name="statistical_summary_all_features.csv",
                        mime="text/csv"
                    )
                else:
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
                        summary_all_df.to_excel(w, index=False, sheet_name="Stat Summary (All)")
                    st.download_button(
                        "⬇️ Download Excel",
                        data=buf.getvalue(),
                        file_name="statistical_summary_all_features.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            # Persist if needed downstream
            st.session_state["statistical_summary_all_df"] = summary_all_df.copy()


    # ---------- Factor Influence ----------
    st.markdown("### 📈 Factor Influence on Target")

    if "positive_factors" not in st.session_state:
        st.session_state.positive_factors = []
    if "negative_factors" not in st.session_state:
        st.session_state.negative_factors = []

    def update_positive():
        st.session_state.negative_factors = [
            f for f in st.session_state.negative_factors if f not in st.session_state.positive_factors
        ]

    def update_negative():
        st.session_state.positive_factors = [
            f for f in st.session_state.positive_factors if f not in st.session_state.negative_factors
        ]

    available_positive = [f for f in factor_cols if f not in st.session_state.negative_factors]
    available_negative = [f for f in factor_cols if f not in st.session_state.positive_factors]

    st.multiselect(
        "✅ Positive Factors (↑ Factor → ↑ Target)",
        options=available_positive,
        key="positive_factors",
        on_change=update_positive
    )

    st.multiselect(
        "❌ Negative Factors (↑ Factor → ↓ Target)",
        options=available_negative,
        key="negative_factors",
        on_change=update_negative
    )

    neutral_factors = [
        f for f in factor_cols if f not in st.session_state.positive_factors + st.session_state.negative_factors
    ]
    if neutral_factors:
        st.info(f"🟡 Neutral/unclassified factors: {', '.join(neutral_factors)}")


    # Show "Finalize" button only if any factor is selected
    if st.session_state.positive_factors or st.session_state.negative_factors:
        if st.button("✅ Finalize Factor Influence"):
            st.session_state["final_positive"] = st.session_state.positive_factors.copy()
            st.session_state["final_negative"] = st.session_state.negative_factors.copy()
            st.session_state["final_neutral"] = neutral_factors.copy()
            st.success("✅ Factor influence finalized.")

            # ---------- Correlation Check ----------
            st.markdown("### 🔍 Validating Factor Directions with Correlation")
            df_corr_check = df_up[[*factor_cols, target_col]].copy()
            correlations = df_corr_check.corr(numeric_only=True)[target_col].drop(target_col)

            removal_reasons = {}

            for f in st.session_state["final_positive"]:
                if correlations.get(f, 0) < 0:
                    removal_reasons[f] = f"❌ Expected positive correlation, got {correlations[f]:.2f}"

            for f in st.session_state["final_negative"]:
                if correlations.get(f, 0) > 0:
                    removal_reasons[f] = f"❌ Expected negative correlation, got {correlations[f]:.2f}"

            st.session_state["final_positive"] = [f for f in st.session_state["final_positive"] if f not in removal_reasons]
            st.session_state["final_negative"] = [f for f in st.session_state["final_negative"] if f not in removal_reasons]
            if removal_reasons:
                st.error("🚫 The following factors were removed due to correlation mismatch:")
                for factor, reason in removal_reasons.items():
                    st.write(f"• **{factor}** — {reason}")
            else:
                st.success("✅ All factors directionally consistent with the target.")

    # ---------- Prepare Final Feature Set ----------
    if (
            "final_positive" in st.session_state or
            "final_negative" in st.session_state or
            "final_neutral" in st.session_state
    ):
        final_feats = (
                st.session_state.get("final_positive", []) +
                st.session_state.get("final_negative", []) +
                st.session_state.get("final_neutral", [])
        )
    else:
        # No correlation check done — fallback to all selected factor columns
        final_feats = st.session_state.get("factor_cols", [])

    # ---------- Feature Selection ----------
    st.markdown("### 🧪 Feature Selection")

    use_lasso = st.checkbox("🔍 Use LASSO Regression")
    use_pval  = st.checkbox("📊 Use Correlation p-value (Pearson, univariate)")

    selected_features = final_feats.copy()

    if not use_lasso and not use_pval:
        st.info("ℹ️ No feature selection method selected. Using all finalized factors.")
    else:
        lasso_feats, pval_feats = [], []

        if use_lasso:
            X = df_up[final_feats]
            y = df_up[target_col]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            lasso = LassoCV(cv=5).fit(X_scaled, y)
            coef = pd.Series(lasso.coef_, index=final_feats)
            lasso_feats = coef[coef != 0].index.tolist()

        if use_pval:
            # Univariate Pearson p-values for each feature vs the target
            df_for_p = df_up[[c for c in final_feats if c in df_up.columns] + [target_col]].copy()
            df_for_p = df_for_p.apply(pd.to_numeric, errors="coerce")

            try:
                from scipy.stats import pearsonr
                _HAS_SCIPY = True
            except Exception:
                _HAS_SCIPY = False

            PVAL_ALPHA = 0.05
            pval_feats = []
            for f in final_feats:
                x = df_for_p[f]
                y = df_for_p[target_col]
                valid = x.notna() & y.notna()
                xv, yv = x[valid], y[valid]

                if len(xv) >= 3 and xv.nunique() > 1 and yv.nunique() > 1:
                    if _HAS_SCIPY:
                        r, p = pearsonr(xv, yv)
                    else:
                        X1 = sm.add_constant(xv)
                        model1 = sm.OLS(yv, X1).fit()
                        p = model1.pvalues.get(f, float("nan"))
                else:
                    p = float("nan")

                if pd.notna(p) and p < PVAL_ALPHA:
                    pval_feats.append(f)

        # Combine based on selections
        if use_lasso and use_pval:
            selected_features = list(set(lasso_feats) & set(pval_feats))
            st.info(f"✅ Using intersection of LASSO and Pearson p-value methods: `{', '.join(selected_features)}`")
        elif use_lasso:
            selected_features = lasso_feats
            st.info(f"✅ Features selected by LASSO: `{', '.join(selected_features)}`")
        elif use_pval:
            selected_features = pval_feats
            st.info(f"✅ Features selected by Pearson correlation p-value: `{', '.join(selected_features)}`")

    st.session_state["final_selected_features"] = selected_features

    # ---------- Final Set Display ----------
    st.markdown("### 🧾 Final Set of Selected Features")

    # Fall back to all features if none were selected by FS methods
    if not selected_features:
        selected_features = final_feats.copy()
        st.session_state["final_selected_features"] = selected_features

    st.success(f"🎯 Final features to be used for modeling: `{', '.join(selected_features)}`")

    # ---------- ➕ Manual Override ----------
    if set(selected_features) != set(final_feats):
        st.markdown("### ➕ Manually Add Features (Override)")

        unselected_feats = [f for f in final_feats if f not in selected_features]

        manual_add = st.multiselect(
            "Select additional features to include:",
            options=unselected_feats,
            key="manual_override_feats"
        )

        if manual_add:
            st.session_state["final_selected_features"] += [
                f for f in manual_add if f not in st.session_state["final_selected_features"]
            ]
            st.success(f"✅ Added manually: `{', '.join(manual_add)}`")

    st.markdown("### 📊 Cleaned Data (Selected Features Only)")
    df_final = st.session_state["uploaded_df"].copy()
    final_selected_features = st.session_state["final_selected_features"]
    geo_unit_col = st.session_state["geo_unit_col"]
    target_col = st.session_state["target_col"]
    mapping = st.session_state["mapping"]

    # Always prepare ungrouped_df
    ungrouped_df = df_final[[geo_unit_col] + final_selected_features + [target_col]].copy()

    # ---------------- Custom per-column aggregation + groupby ----------------
    # Candidates to aggregate: selected features + target (exclude the geo key)
    agg_candidates = [c for c in ([*final_selected_features, target_col]) if c != geo_unit_col and c in ungrouped_df.columns]

    # Build default choices based on dtype
    numeric_set = set(ungrouped_df.select_dtypes(include="number").columns)
    default_map = {}
    for c in agg_candidates:
        default_map[c] = "mean" if c in numeric_set else "mode"

    # Persist/restore user choices
    if "agg_map" not in st.session_state:
        st.session_state.agg_map = default_map.copy()
    else:
        # keep keys in sync with current candidates
        for c in agg_candidates:
            st.session_state.agg_map.setdefault(c, default_map[c])
        for k in list(st.session_state.agg_map.keys()):
            if k not in agg_candidates:
                st.session_state.agg_map.pop(k, None)

    st.markdown("#### Grouping & Aggregation")
    with st.form("agg_chooser_form"):
        # Render a compact two-column selector UI
        a1, a2 = st.columns([2, 2])
        # To keep layout tidy, split the list roughly in half
        mid = max(1, len(agg_candidates)//2)
        left_cols  = agg_candidates[:mid]
        right_cols = agg_candidates[mid:]

        # Allowed aggregations
        NUMERIC_AGGS = ["mean", "sum", "median", "min", "max", "std", "count", "nunique"]
        GENERIC_AGGS = ["mode", "first", "last", "count", "nunique"]

        def _agg_picker(container, cols):
            for c in cols:
                opts = NUMERIC_AGGS if c in numeric_set else GENERIC_AGGS
                st.session_state.agg_map[c] = container.selectbox(
                    f"{c}",
                    options=opts,
                    index=opts.index(st.session_state.agg_map.get(c, opts[0])),
                    key=f"agg_sel::{c}"
                )

        with a1:
            _agg_picker(st, left_cols)
        with a2:
            _agg_picker(st, right_cols)

        apply_clicked = st.form_submit_button("Apply aggregation & build grouped view", type="primary")

    # Helper for mode that returns a single value
    def _first_mode(s: pd.Series):
        m = s.mode(dropna=True)
        return m.iloc[0] if not m.empty else (np.nan if s.dtype.kind in "fc" else None)

    # Build the grouped df if user applied or if it doesn't exist yet
    if (apply_clicked or "grouped_df" not in st.session_state) and geo_unit_col in df_final.columns:
        # Construct a dict col -> callable/string understood by pandas
        agg_map = {}
        for c, fn in st.session_state.agg_map.items():
            if fn == "mode":
                agg_map[c] = _first_mode
            else:
                agg_map[c] = fn  # pandas supports these strings directly

        # Only aggregate columns that are present (defensive)
        valid_agg_map = {c: fn for c, fn in agg_map.items() if c in ungrouped_df.columns}

        try:
            grouped_df = (
                ungrouped_df
                .groupby(geo_unit_col, dropna=False)
                .agg(valid_agg_map)
                .reset_index()
            )
            grouped_df[geo_unit_col] = grouped_df[geo_unit_col].map(mapping)
            st.session_state["grouped_df"] = grouped_df
            st.success("Grouped view updated with selected aggregations.")
        except Exception as e:
            st.error("Failed to build grouped table with the chosen aggregations.")
            st.exception(e)


    # Determine what to show based on grouping availability
    if "grouped_df" in st.session_state  and len(st.session_state["grouped_df"]) < len(ungrouped_df):
        view_option = st.selectbox("Choose data view:", ["Grouped (Average by Geo Unit)", "Ungrouped (Raw Records)"])
        if view_option == "Grouped (Average by Geo Unit)":
            df_to_display = st.session_state["grouped_df"]
        else:
            df_to_display = ungrouped_df
    else:
        df_to_display = ungrouped_df

    # Display table
    st.dataframe(df_to_display.reset_index(drop=True), use_container_width=True)

    # Download section
    with st.expander("📤 Select Download Format"):
        format_choice = st.radio("Choose format:", ["CSV", "Excel"], horizontal=True)

        if format_choice == "CSV":
            csv = df_to_display.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
        elif format_choice == "Excel":
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df_to_display.to_excel(writer, index=False, sheet_name="Cleaned Data")
            st.download_button(
                label="⬇️ Download Excel",
                data=excel_buffer.getvalue(),
                file_name="cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )



    # ---------- Drag & Drop Feature Bucketing (Kanban style) ----------
    st.markdown("## 🧺 Feature Buckets (Drag & Drop)")

    # Try to use streamlit-sortables for a nice drag & drop UI.
    try:
        from streamlit_sortables import sortables  # pip install streamlit-sortables
        _HAS_SORTABLES = True
    except Exception:
        _HAS_SORTABLES = False

    # Pull selected features
    final_feats_for_bucket = st.session_state.get("final_selected_features", [])

    if not final_feats_for_bucket:
        st.info("No selected features available to bucket yet.")
    else:
        # ---------- defaults & state ----------
        DEFAULT_BUCKETS = ["Demographics", "Infrastructure", "Linguistics", "Economics"]
        if "buckets" not in st.session_state:
            st.session_state.buckets = DEFAULT_BUCKETS.copy()

        # Keep a stable pool of all features
        if "all_feats_pool" not in st.session_state:
            st.session_state.all_feats_pool = sorted(set(final_feats_for_bucket))
        else:
            # Sync in case features changed upstream
            for f in final_feats_for_bucket:
                if f not in st.session_state.all_feats_pool:
                    st.session_state.all_feats_pool.append(f)
            st.session_state.all_feats_pool = [f for f in st.session_state.all_feats_pool if f in final_feats_for_bucket]
            st.session_state.all_feats_pool.sort()

        # Each bucket holds a list of assigned features; '(Pool)' holds unassigned
        if "bucket_lists" not in st.session_state:
            st.session_state.bucket_lists = {"(Pool)": st.session_state.all_feats_pool.copy()}
            for b in st.session_state.buckets:
                st.session_state.bucket_lists.setdefault(b, [])

        # Ensure state integrity when buckets change upstream
        # 1) Ensure all current buckets exist in dict
        for b in st.session_state.buckets:
            st.session_state.bucket_lists.setdefault(b, [])
        # 2) Remove stray bucket keys not in the current bucket list (except Pool)
        for k in list(st.session_state.bucket_lists.keys()):
            if k not in st.session_state.buckets and k != "(Pool)":
                st.session_state.bucket_lists["(Pool)"].extend(st.session_state.bucket_lists[k])
                st.session_state.bucket_lists.pop(k, None)
        # 3) Ensure every feature appears in exactly one list (deduplicate + refill)
        seen = set()
        for k in list(st.session_state.bucket_lists.keys()):
            unique = []
            for f in st.session_state.bucket_lists[k]:
                if f in final_feats_for_bucket and f not in seen:
                    unique.append(f); seen.add(f)
            st.session_state.bucket_lists[k] = unique
        # Put any missing features back to pool
        missing_back_to_pool = [f for f in final_feats_for_bucket if f not in seen]
        for f in missing_back_to_pool:
            st.session_state.bucket_lists["(Pool)"].append(f)
        st.session_state.bucket_lists["(Pool)"].sort()

        # ---------- bucket management UI (forms to avoid session_state write-after-create) ----------
        st.markdown("#### Buckets")
        bcol1, _, bcol3 = st.columns([4, 1, 3])

        # Add bucket (clears input automatically on submit)
        with bcol1:
            with st.form("bucket_add_form", clear_on_submit=True):
                new_bucket = st.text_input(
                    "➕ Add a new bucket",
                    placeholder="e.g., Health, Education",
                    key="__new_bucket_name",
                )
                add_clicked = st.form_submit_button("Add Bucket")

            if add_clicked:
                name = (new_bucket or "").strip()
                if not name:
                    st.warning("Bucket name cannot be empty.")
                elif name in st.session_state.buckets:
                    st.warning(f"Bucket '{name}' already exists.")
                else:
                    st.session_state.buckets.append(name)
                    st.session_state.bucket_lists.setdefault(name, [])
                    st.success(f"Bucket '{name}' added.")

        # Remove bucket (separate form to isolate widget state)
        with bcol3:
            if st.session_state.buckets:
                with st.form("bucket_remove_form"):
                    to_remove = st.selectbox(
                        "🗑️ Remove a bucket",
                        ["(Choose)"] + st.session_state.buckets,
                        key="__rm_bucket_sel",
                        )
                    rm_clicked = st.form_submit_button("Remove Selected")

                if rm_clicked and to_remove and to_remove != "(Choose)":
                    st.session_state.bucket_lists["(Pool)"].extend(
                        st.session_state.bucket_lists.get(to_remove, [])
                    )
                    st.session_state.bucket_lists.pop(to_remove, None)
                    st.session_state.buckets = [b for b in st.session_state.buckets if b != to_remove]
                    st.success(f"Removed bucket '{to_remove}' and returned its features to the pool.")

        # ---------- DRAG & DROP UI ----------
        st.markdown("#### Assign features by dragging from the **Pool** to your buckets")

        if _HAS_SORTABLES:
            # Compose the list-of-lists in left-to-right order: Pool + buckets
            labels = ["(Pool)"] + st.session_state.buckets
            lists = [st.session_state.bucket_lists[label] for label in labels]

            # Render interactive sortables; returns reordered lists in same structure
            new_lists = sortables(
                lists,
                labels=labels,
                direction='vertical',
                key="feature_bucket_sortables",
                drag=True
            )

            # Update state from result
            for idx, label in enumerate(labels):
                seen_local = set()
                cleaned = []
                for f in new_lists[idx]:
                    if f in final_feats_for_bucket and f not in seen_local:
                        cleaned.append(f); seen_local.add(f)
                st.session_state.bucket_lists[label] = cleaned

            # Ensure uniqueness across all lists
            seen_global = set()
            for label in labels:
                unique = []
                for f in st.session_state.bucket_lists[label]:
                    if f not in seen_global:
                        unique.append(f); seen_global.add(f)
                st.session_state.bucket_lists[label] = unique
            for f in final_feats_for_bucket:
                if f not in seen_global:
                    st.session_state.bucket_lists["(Pool)"].append(f)

        else:
            # Fallback: simple selectbox UI (no external dependency)
            if "feature_bucket_map" not in st.session_state:
                st.session_state.feature_bucket_map = {f: "(Unassigned)" for f in final_feats_for_bucket}
            else:
                for f in final_feats_for_bucket:
                    st.session_state.feature_bucket_map.setdefault(f, "(Unassigned)")
                for f in list(st.session_state.feature_bucket_map.keys()):
                    if f not in final_feats_for_bucket:
                        st.session_state.feature_bucket_map.pop(f, None)

            choices = ["(Unassigned)"] + st.session_state.buckets
            for f in sorted(final_feats_for_bucket):
                st.session_state.feature_bucket_map[f] = st.selectbox(
                    f"• {f}",
                    options=choices,
                    index=choices.index(st.session_state.feature_bucket_map.get(f, "(Unassigned)")),
                    key=f"fallback_bucket_sel_{f}"
                )
            # Build bucket_lists from mapping
            st.session_state.bucket_lists = {"(Pool)": [], **{b: [] for b in st.session_state.buckets}}
            for f, b in st.session_state.feature_bucket_map.items():
                if b == "(Unassigned)":
                    st.session_state.bucket_lists["(Pool)"].append(f)
                else:
                    st.session_state.bucket_lists[b].append(f)

        # ---------- Summary, validation & download ----------
        # Build mapping df (feature -> bucket), pool counted as "(Unassigned)"
        mapping_records = []
        for b, items in st.session_state.bucket_lists.items():
            if b == "(Pool)":
                for f in items:
                    mapping_records.append({"feature": f, "bucket": "(Unassigned)"})
            else:
                for f in items:
                    mapping_records.append({"feature": f, "bucket": b})
        mapping_df = pd.DataFrame(sorted(mapping_records, key=lambda x: (x["bucket"], x["feature"])))

        # Counts
        counts = mapping_df.value_counts("bucket").to_frame("count").reset_index()

        st.markdown("#### 📋 Bucket Summary")
        st.dataframe(mapping_df, use_container_width=True)
        st.markdown("**Counts by bucket**")
        st.dataframe(counts, use_container_width=True)

        # Warn if anything still unassigned
        unassigned = mapping_df.loc[mapping_df["bucket"] == "(Unassigned)", "feature"].tolist()
        if unassigned:
            st.warning(f"Unassigned features: {', '.join(unassigned)}")

        # Persist for downstream usage
        grouped = (
            mapping_df[mapping_df["bucket"] != "(Unassigned)"]
            .groupby("bucket")["feature"].apply(list).to_dict()
        )
        st.session_state["feature_buckets_grouped"] = grouped
        st.session_state["feature_bucket_mapping_df"] = mapping_df.copy()

        with st.expander("📤 Download Feature→Bucket Mapping"):
            fmt_choice = st.radio("Choose format:", ["CSV", "Excel"], horizontal=True, key="bucket_dl_fmt_drag")
            if fmt_choice == "CSV":
                csv_bytes = mapping_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_bytes,
                    file_name="feature_bucket_mapping.csv",
                    mime="text/csv"
                )
            else:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                    mapping_df.to_excel(writer, index=False, sheet_name="Feature Buckets")
                st.download_button(
                    label="⬇️ Download Excel",
                    data=excel_buffer.getvalue(),
                    file_name="feature_bucket_mapping.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
