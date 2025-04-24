import streamlit as st

# Current app version
APP_VERSION = "2025-04-24.1"
APP_VERSION_LABEL = "without Animation"

def setup_page_config():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="1D Clifford QCA Simulator",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        color: #4B0082;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .description {
        color: #555;
        font-size: 1.0rem;
        margin-bottom: 2rem;
    }
    .sidebar-header {
        color: #4B0082;
        font-weight: 600;
        margin-top: 1rem;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #eee;
        padding: 1rem;
    }
    .stAlert {
        padding: 0.75rem;
    }
    </style>
    """, unsafe_allow_html=True)

def display_app_version():
    """Display the app version at the bottom of the sidebar."""
    st.sidebar.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"**App Version: {APP_VERSION} ({APP_VERSION_LABEL})**") 