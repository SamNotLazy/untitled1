# utils/data_prep.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro

# ---------- Fuzzy Matching ----------
try:
    from rapidfuzz import process, fuzz
    def fuzzy_map(upload_names, valid, thresh=80):
        rows, mp, un = [], {}, []
        for name in upload_names:
            best, score, _ = process.extractOne(name, valid, scorer=fuzz.ratio)
            rows.append((name, best if score >= thresh else None, score))
            if score >= thresh:
                mp[name] = best
            else:
                un.append(name)
        return mp, un, pd.DataFrame(
            rows, columns=["Uploaded Name", "Matched Name", "Similarity (%)"]
        )
except ImportError:
    print("RapidFuzz not found, falling back to difflib")
    from difflib import SequenceMatcher
    def fuzzy_map(upload_names, valid, thresh=80):
        rows, mp, un = [], {}, []
        for name in upload_names:
            best, score = None, 0
            for v in valid:
                s = SequenceMatcher(None, name, v).ratio() * 100
                if s > score:
                    best, score = v, s
            rows.append((name, best if score >= thresh else None, score))
            if score >= thresh:
                mp[name] = best
            else:
                un.append(name)
        return mp, un, pd.DataFrame(
            rows, columns=["Uploaded Name", "Matched Name", "Similarity (%)"]
        )


# ---------- Data Cleaning ----------
def clean_df(df):
    df = df.copy()
    # Drop columns that are completely NaN
    df = df.dropna(axis=1, how="all")

    # Fill NaNs with median
    df = df.fillna(df.median(numeric_only=True))

    # Cap outliers using IQR method
    for c in df.select_dtypes("number").columns:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[c] = df[c].clip(lower=lower, upper=upper)
    return df



# # ---------- Linear Regression Assumptions ----------
def check_linear_regression_assumptions(X_train_scaled, y_train, final_features):
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.diagnostic import het_breuschpagan
    from scipy.stats import shapiro

    # Add intercept
    X_train_sm = sm.add_constant(X_train_scaled)
    ols_model = sm.OLS(y_train, X_train_sm).fit()
    residuals = ols_model.resid

    # 1. VIF
    vif_vals = [variance_inflation_factor(X_train_scaled, i) for i in range(X_train_scaled.shape[1])]
    max_vif = max(vif_vals)
    vif_pass = max_vif <= 5

    # 2. Durbin-Watson
    dw_stat = sm.stats.durbin_watson(residuals)
    dw_pass = 1.5 <= dw_stat <= 2.5

    # 3. Homoscedasticity
    _, pval, _, _ = het_breuschpagan(residuals, X_train_sm)
    homo_pass = pval > 0.05

    # 4. Normality
    _, shapiro_p = shapiro(residuals)
    normality_pass = shapiro_p > 0.05

    assumption_report = {
        "Multicollinearity (VIF ≤ 5)": {
            "status": "pass" if vif_pass else "fail",
            "statistic": f"Max VIF = {max_vif:.2f}"
        },
        "Independence (Durbin-Watson ∈ [1.5, 2.5])": {
            "status": "pass" if dw_pass else "fail",
            "statistic": f"DW = {dw_stat:.2f}"
        },
        "Homoscedasticity (p > 0.05)": {
            "status": "pass" if homo_pass else "fail",
            "statistic": f"p = {pval:.4f}"
        },
        "Normality of Residuals (p > 0.05)": {
            "status": "pass" if normality_pass else "fail",
            "statistic": f"p = {shapiro_p:.4f}"
        }
    }

    overall_pass = all(item["status"] == "pass" for item in assumption_report.values())
    return overall_pass, assumption_report





# ---------- Model Fitting ----------
def fit_model(df, target_col, feature_cols, custom_params={}):
    X = df[feature_cols]
    y = df[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {"importances": {}, "models": {}, "scaler": scaler}

    # Linear Regression
    linreg = LinearRegression().fit(X_scaled, y)
    results["importances"]["linear"] = pd.Series(abs(linreg.coef_), index=feature_cols)
    results["models"]["linear"] = linreg

    # Random Forest
    rf = RandomForestRegressor(random_state=42, **custom_params.get("rf", {}))
    rf.fit(X, y)
    results["importances"]["rf"] = pd.Series(rf.feature_importances_, index=feature_cols)
    results["models"]["rf"] = rf

    # XGBoost
    xgb = XGBRegressor(random_state=42, verbosity=0, **custom_params.get("xgb", {}))
    xgb.fit(X, y)
    results["importances"]["xgb"] = pd.Series(xgb.feature_importances_, index=feature_cols)
    results["models"]["xgb"] = xgb

    results["feature_names"] = feature_cols.copy()
    return results
