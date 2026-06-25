"""
app.py (root entry point)
--------------------------
Thin wrapper so `streamlit run app.py` works from the project root, as
referenced in the project structure / deployment instructions (Streamlit
Community Cloud expects an app.py at the repo root).

The actual dashboard implementation lives in dashboard/app.py so that the
`dashboard/` folder stays self-contained and importable on its own.
"""

import runpy
from pathlib import Path

dashboard_app = Path(__file__).resolve().parent / "dashboard" / "app.py"
runpy.run_path(str(dashboard_app), run_name="__main__")
