"""Test bootstrap: put Model/ and tests/ on sys.path.

Run the fast suite with plain `pytest` (slow marker excluded by
pytest.ini addopts); run the artifact-dependent golden tests with
`pytest -m slow`.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "Model"))
sys.path.insert(0, str(_HERE))
