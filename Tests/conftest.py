"""Test bootstrap: put Model/ and tests/ on sys.path, and load Tools
scripts (digit-and-space filenames are not importable) via load_tool.

Run the fast suite with plain `pytest` (slow marker excluded by
pytest.ini addopts); run the artifact-dependent golden tests with
`pytest -m slow`.
"""
import importlib
import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "Model"))
sys.path.insert(0, str(_HERE))

_TOOLS = _HERE.parent / "Tools"
_TOOL_CACHE = {}


def load_tool(stem):
    """Import a Tools/ script by its file stem (e.g. '4) Grade Results').

    Tools filenames start with a digit and contain spaces, so a plain
    `import` is a syntax error; this loads them by path, cached so
    repeated loads across test files share one module instance. Tools/2
    itself imports '1) Get Todays Games' by module NAME, so Tools/ goes
    on sys.path and modules register under their real names."""
    if stem in _TOOL_CACHE:
        return _TOOL_CACHE[stem]
    if str(_TOOLS) not in sys.path:
        sys.path.insert(0, str(_TOOLS))
    mod = importlib.import_module(stem)
    _TOOL_CACHE[stem] = mod
    return mod
