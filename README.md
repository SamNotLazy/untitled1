# Intervention Dashboard (Streamlit)

A multi-page Streamlit app for **data-driven interventions**: upload district-level data, select/validate factors, train models, simulate “what-if” interventions, **prescribe** factor shifts to hit targets, and allocate **budgets** to schemes and beneficiaries across districts.

## ✨ What you can do

- **Upload & validate data** (auto fuzzy-match district names to your shapefile).
- **Select factors** and auto-check directional consistency with correlations.
- **Feature selection** via LASSO and/or OLS p-values.
- **Train/evaluate models** (Linear, Random Forest, XGBoost) with assumption checks.
- **Simulate interventions** (% changes + sensitivities) & map their impact.
- **Prescribe factor adjustments** to reach a target safely within bounds.
- **Allocate budget** by schemes & districts (inverse or direct importance logic).
- **Estimate beneficiaries** using unit costs (₹ per beneficiary).

---

## 🗂️ Project structure

```
.
├─ app.py
├─ pages/
│  ├─ home.py
│  ├─ intervention_setup.py
│  ├─ differential_impact.py
│  ├─ prescriptive_modelling.py
│  ├─ budget_allocation.py
│  ├─ schemes_dashboard.py
│  └─ beneficiaries_dashboard.py
├─ utils/
│  ├─ data_prep.py
│  ├─ viz.py
│  └─ budget_utils.py        # (expected; see note below)
└─ States/
   └─ <STATE>/
      └─ <STATE>_DISTRICTS.geojson   # must include a 'dtname' column
```



---

## 🚀 Quickstart

### 1) Environment

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies

```
streamlit
pandas
geopandas
shapely
fiona
pyproj
plotly
statsmodels
scikit-learn
xgboost
scipy
rapidfuzz
openpyxl
XlsxWriter
```

### 3) Provide shapefiles

Place your state boundaries as GeoJSON:

```
States/<STATE>/<STATE>_DISTRICTS.geojson
```

### 4) Run

```bash
streamlit run app.py
```

---

## 📥 Expected input data

Upload a CSV/Excel with at least:

- **First column**: Geo unit (e.g., District names) — will be fuzzy-matched to shapefile (`dtname`)
- **One target column**: outcome you want to improve
- **Factor columns**: independent variables (features)

---

## 🧭 App workflow (pages)

- **Home** → State & data upload, fuzzy matching, factor selection, feature selection
- **Intervention Setup** → Bounds, schemes, sensitivities, model training, assumption checks
- **Differential Impact Analysis** → What-if analysis, maps & tables
- **Prescriptive Modelling** → Target-driven factor adjustments
- **Budget Allocation** → Scheme & district budget allocation
- **Schemes Dashboard** → Scheme-wise budget distribution
- **Beneficiaries Dashboard** → Scheme-wise beneficiary estimation

---

## 🧰 Utilities

### utils/data_prep.py
- Fuzzy matching with RapidFuzz/difflib
- Data cleaning & outlier clipping
- Linear regression assumption checks (VIF, DW, BP, Shapiro)
- Model fitting (Linear, RF, XGB)

### utils/viz.py
- Plot maps + bar charts side-by-side with Plotly

### utils/budget_utils.py (expected)
- `compute_budget_allocation()` for budget splits

---

## 📦 Example `requirements.txt`

```txt
streamlit
pandas
geopandas
shapely
fiona
pyproj
plotly
statsmodels
scikit-learn
xgboost
scipy
rapidfuzz
openpyxl
XlsxWriter
```

---

## 📜 License

MIT 
