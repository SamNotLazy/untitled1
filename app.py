import sys
import importlib
import traceback
import streamlit as st

from pages.public_dashboard_view import run_public_dashboard_view
from pages.home import run_home
from pages.intervention_setup import run_intervention_setup
from pages.differential_impact import run_differential_impact_analysis
from pages.prescriptive_modelling import run_prescriptive_modelling
from pages.budget_allocation import run_budget_allocation
from pages.private_dashboard_gallery import run_private_dashboard_gallery


# --- USERNAME AND PASSWORD MANAGEMENT ---
# NOTE: For a real application, you should use a secure authentication method
# like Firebase Authentication, Auth0, or connect to a database.
# Hardcoding credentials is for demonstration purposes only.
CORRECT_USERNAME = ""
CORRECT_PASSWORD = ""


st.set_page_config(page_title="Intervention Dashboard", layout="wide")

# ---------------- CSS ----------------
st.markdown(
    """
    <style>

    /* Remove default multi-page Streamlit menu */
    [data-testid="stSidebarNav"] {display: none;}

    .main > div:first-child { padding-top: 0rem !important; }
    .main .block-container { padding: 0 2rem 1rem 2rem !important; }
    [data-testid="stSidebar"] { width: 220px; }
    .css-1cpxqw2, .stSlider { font-size: 0.85rem; }

    body { overflow-x: auto !important; max-width: 100vw !important; }

    /* Prevent layout shift on rerender */
    .block-container {
        transition: none !important;
        animation: none !important;
    }
    .stPlotlyChart {
        transition: none !important;
        animation: none !impoartant;
    }

    </style>
    """,
    unsafe_allow_html=True,
)



# --- LOGIN/LOGOUT LOGIC ---
# A session state variable tracks if the user is logged in.
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "pre_login_page" not in st.session_state:
    st.session_state.pre_login_page = "Login"
if "public_sub_page" not in st.session_state:
    st.session_state.public_sub_page = "Public Dashboard Gallery"
if "logged_in_main_section" not in st.session_state:
    st.session_state.logged_in_main_section = "Home"
if "logged_in_sub_page" not in st.session_state:
    st.session_state.logged_in_sub_page = "Intervention Setup"

def login():
    """Handles the login form and checks credentials."""
    st.header("Login")
    st.markdown("---")
    with st.form(key='login_form'):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

    if login_button:
        if username == CORRECT_USERNAME and password == CORRECT_PASSWORD:
            st.session_state.clear()
            st.session_state["logged_in"] = True
            st.success("Login successful!")
            st.rerun()  # Rerun the app to show the dashboard
        else:
            st.error("Incorrect username or password.")

def logout():
    """Logs out the user by resetting the session state."""
    st.session_state.logged_in = False
    st.session_state.pre_login_page = "Public Dashboards"
    st.session_state.public_sub_page = "Public Dashboard Gallery"
    st.rerun() # Rerun the app to show the login screen

# If the user is not logged in, show the login page or public dashboards.
if not st.session_state.logged_in:
    st.sidebar.title("Navigation")
    if st.sidebar.button("Login"):
        st.session_state.pre_login_page = "Login"
    if st.sidebar.button("Public Dashboards"):
        st.session_state.pre_login_page = "Public Dashboards"

    if st.session_state.pre_login_page == "Login":
        login()
        st.info("Please click or check the public dashboards tab for loading a public dashboard in order to see the visualizations")
    elif st.session_state.pre_login_page == "Public Dashboards":
        run_public_dashboard_view()
        if "dashboard_loaded" in st.session_state:
            st.sidebar.title(st.session_state["loaded_dashboard_name"])
            if st.sidebar.button("Differential Impact Analysis"):
                st.session_state["public_sub_page"] = "Differential Impact Analysis"
                st.session_state.pre_login_page = "Sub-Page"
                st.rerun()
            if st.sidebar.button("Prescriptive Modelling"):
                st.session_state["public_sub_page"] = "Prescriptive Modelling"
                st.session_state.pre_login_page = "Sub-Page"
                st.rerun()

            if st.sidebar.button("Budget Allocation"):
                st.session_state["public_sub_page"] = "Budget Allocation"
                st.session_state.pre_login_page = "Sub-Page"
                st.rerun()





    else:
        st.sidebar.title(st.session_state["loaded_dashboard_name"])
        if st.sidebar.button("Differential Impact Analysis"):
            st.session_state["public_sub_page"] = "Differential Impact Analysis"
            st.session_state.pre_login_page = "Sub-Page"
        if st.sidebar.button("Prescriptive Modelling"):
            st.session_state["public_sub_page"] = "Prescriptive Modelling"
            st.session_state.pre_login_page = "Sub-Page"
        if st.sidebar.button("Budget Allocation"):
            st.session_state["public_sub_page"] = "Budget Allocation"
            st.session_state.pre_login_page = "Sub-Page"



        if st.session_state["public_sub_page"] == "Differential Impact Analysis":
            run_differential_impact_analysis()
        elif st.session_state["public_sub_page"] == "Prescriptive Modelling":
            run_prescriptive_modelling()
        elif st.session_state["public_sub_page"] == "Budget Allocation":
            run_budget_allocation()





else:
    # --- SIDEBAR NAVIGATION FOR LOGGED IN USERS ---
    # st.sidebar.title("Navigation")
    st.sidebar.button("Logout", on_click=logout)

    main_section = st.sidebar.radio("Main Sections", [
        "Home",
        "Intervention Dashboard",
        "More Analytics",
        "Dashboard Gallery"
    ])

    # --- PAGE HANDLING ---
    page = ""

    if main_section == "Home":
        run_home()

    elif main_section == "Intervention Dashboard":
        sub_page = st.sidebar.radio("Dashboard Options", [
            "Intervention Setup",
            "Differential Impact Analysis",
            "Prescriptive Modelling",
            "Budget Allocation",
        ])
        if sub_page == "Intervention Setup":
            run_intervention_setup()
        elif sub_page == "Differential Impact Analysis":
            run_differential_impact_analysis()
        elif sub_page == "Prescriptive Modelling":
            run_prescriptive_modelling()
        elif sub_page == "Budget Allocation":
            run_budget_allocation()

    elif main_section == "More Analytics":
        st.title("📈 More Analytics")
        st.write("Advanced visualizations and comparisons.")

    elif main_section == "Public Dashboards":
        sub_page = st.sidebar.radio("Public Subsections", [
            "Public Dashboard Gallery",
            "Differential Impact Analysis",
            "Prescriptive Modelling",
            "Budget Allocation",
        ])
        if sub_page == "Public Dashboard Gallery":
            run_public_dashboard_view()
        elif sub_page == "Differential Impact Analysis":
            run_differential_impact_analysis()
        elif sub_page == "Prescriptive Modelling":
            run_prescriptive_modelling()
        elif sub_page == "Budget Allocation":
            run_budget_allocation()

    elif main_section == "Dashboard Gallery":
        run_private_dashboard_gallery()
