

import pandas as pd
import streamlit as st


def compute_budget_allocation(df, feature_importances, total_budget, geo_unit_col, mode="inverse"):
    """
    Distribute a total budget across features and then across districts.

    Parameters:
        df (pd.DataFrame): DataFrame containing feature values.
        feature_importances (dict): Mapping of features to importance values.
        total_budget (float): Total budget (in Cr).
        geo_unit_col (str): Column name representing geographic units (e.g., District).
        mode (str): Allocation mode, "inverse" or "direct".

    Returns:
        feature_budgets (pd.Series): Budget allocated to each feature.
        district_allocations (pd.DataFrame): District-wise allocations for each feature and total.
    """
    # --- Step 0: Retrieve positive/negative indicators and target direction ---
    positive_indicators = st.session_state.get("final_positive", [])
    negative_indicators = st.session_state.get("final_negative", [])
    direction = st.session_state.get("target_direction", "Increase")

    # --- Step 1: Compute Feature Weights ---
    fi_series = pd.Series(feature_importances).abs()
    weights = 1 / fi_series if mode == "inverse" else fi_series
    feature_weights = weights / weights.sum()

    # --- Step 2: Allocate Budget to Each Feature ---
    feature_budgets = feature_weights * total_budget

    # --- Step 3: Allocate Feature Budgets Across Districts ---
    district_allocations = pd.DataFrame(index=df.index)
    for feat in feature_budgets.index:
        values = df[feat].clip(lower=0)  # Avoid negative values
        total_feat_value = values.sum()

        if total_feat_value == 0:
            allocations = 0
        else:
            # Default proportional allocation
            alloc_values = values

            # Invert allocation for certain indicators based on direction
            if (
                (direction.lower() == "increase" and feat in positive_indicators)
                or (direction.lower() == "decrease" and feat in negative_indicators)
            ):
                alloc_values = total_feat_value - values

            total_alloc_value = alloc_values.sum()
            allocations = (
                (alloc_values / total_alloc_value) * feature_budgets[feat]
                if total_alloc_value != 0
                else 0
            )

        district_allocations[feat] = allocations

    # --- Step 4: Compute Total Budget per District ---
    district_allocations["Total_Budget"] = district_allocations.sum(axis=1)

    # --- Step 5: Add Geo Unit Column for Labels ---
    if geo_unit_col in df.columns:
        district_allocations.insert(0, geo_unit_col, df[geo_unit_col])

    return feature_budgets, district_allocations





