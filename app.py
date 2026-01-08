"""Streamlit Cloud entry point for EPL Bet Indicator."""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.ui.dashboard import main

if __name__ == "__main__":
    main()
