"""EPL Bet Indicator Dashboard - Single Page Layout."""
import streamlit as st

# Load Streamlit secrets into env vars BEFORE importing other modules
from src.config import load_streamlit_secrets
load_streamlit_secrets()

from src.storage.database import db

# Ensure database tables exist on startup
db.create_tables()

st.set_page_config(
    page_title="EPL Bet Indicator",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for professional dark mode styling
CUSTOM_CSS = """
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Open+Sans:wght@400;500;600&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Open Sans', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
    }

    /* Main header gradient */
    .main h1 {
        background: linear-gradient(135deg, #F59E0B 0%, #FBBF24 50%, #F59E0B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700 !important;
    }

    /* Hide sidebar */
    section[data-testid="stSidebar"] {
        display: none;
    }

    /* Card styling */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
    }

    /* Metric cards enhancement */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        transition: all 0.3s ease;
    }

    div[data-testid="stMetric"]:hover {
        border-color: #F59E0B;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.15);
    }

    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #0F172A;
        padding: 0.5rem;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #94A3B8;
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        padding: 0.5rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%) !important;
        color: #0F172A !important;
    }

    /* Match selector styling */
    .match-selector-container {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* Expander styling */
    details[data-testid="stExpander"] {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
    }

    details[data-testid="stExpander"] summary {
        color: #F8FAFC;
        font-weight: 500;
    }

    details[data-testid="stExpander"][open] {
        border-color: #F59E0B;
    }

    /* Select boxes and sliders */
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stSlider"] > div {
        background: transparent;
    }

    /* Divider styling */
    hr {
        border-color: #334155 !important;
        margin: 1.5rem 0 !important;
    }

    /* Info/Warning boxes */
    div[data-testid="stAlert"] {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: #0F172A;
        border: none;
        border-radius: 8px;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #FBBF24 0%, #F59E0B 100%);
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
        transform: translateY(-1px);
    }

    /* Data editor / tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #334155;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Caption text */
    .stCaption {
        color: #64748B !important;
    }

    /* Signal card styling */
    .signal-card {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    .signal-card:hover {
        border-color: #F59E0B;
        box-shadow: 0 8px 30px rgba(245, 158, 11, 0.1);
    }

    .signal-card.edge-strong {
        border-left: 4px solid #10B981;
    }

    .signal-card.edge-moderate {
        border-left: 4px solid #F59E0B;
    }

    .signal-card.edge-marginal {
        border-left: 4px solid #FB923C;
    }

    /* Edge badge colors */
    .edge-strong { color: #10B981; }
    .edge-moderate { color: #F59E0B; }
    .edge-marginal { color: #FB923C; }
    .edge-below { color: #64748B; }

    /* Section headers */
    .section-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: #F8FAFC;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #334155;
    }

    /* Quick stats row */
    .quick-stats-container {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    }

    /* Footer branding */
    footer {
        visibility: hidden;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0F172A;
    }

    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }
</style>
"""


def main():
    """Main dashboard entry - single page layout."""
    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.title("EPL Bet Indicator")
    st.caption("Multi-market betting analytics for Polymarket")

    # Import and render the unified match analysis (main content)
    from src.ui.pages import match_analysis_unified
    match_analysis_unified.render()

    # Collapsible Historical Performance section
    st.markdown("<div class='section-header'>Historical Performance</div>", unsafe_allow_html=True)
    with st.expander("View Historical Analytics", expanded=False):
        from src.ui.pages import historical
        historical.render()

    # Collapsible Settings section
    st.markdown("<div class='section-header'>Settings</div>", unsafe_allow_html=True)
    with st.expander("Configure Settings", expanded=False):
        from src.ui.pages import settings_page
        settings_page.render()


if __name__ == "__main__":
    main()
