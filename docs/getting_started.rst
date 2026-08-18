Getting started
===============

Install the runtime dependencies from the repository root::

   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

Run the interactive example from the repository root so its relative model and
map paths resolve correctly::

   python main.py

The first run compiles the generated CasADi C functions. See the project
``README.md`` for system packages, configuration, and troubleshooting details.

Building these pages
--------------------

Install the documentation dependencies and build HTML output::

   python -m pip install -r docs/requirements.txt
   sphinx-build -M html docs docs/_build
