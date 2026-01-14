from pathlib import Path
import subprocess
import sys

app = Path(__file__).resolve().parents[1] / "src" / "chexpert_mvp" / "ui" / "app_streamlit.py"
subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], check=False)
