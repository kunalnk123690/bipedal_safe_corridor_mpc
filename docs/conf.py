"""Sphinx configuration for the safe-corridor footstep planner."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

project = "Safe-Corridor Footstep Planner for Digit"
author = "Kunal S. Narkhede, Abhijeet M. Kulkarni, Dhruv A. Thanki, and Ioannis Poulakakis"
copyright = "2022, Kunal S. Narkhede, Abhijeet M. Kulkarni, Dhruv A. Thanki, and Ioannis Poulakakis"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_default_options = {
    "members": True,
    "private-members": True,
    "special-members": "__init__",
    "show-inheritance": True,
}
autodoc_mock_imports = ["casadi", "glfw", "mujoco", "pyquaternion", "scipy"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_static_path = []
