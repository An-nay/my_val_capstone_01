import sys
from pathlib import Path

# make the regression package importable when pytest is started from any folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
