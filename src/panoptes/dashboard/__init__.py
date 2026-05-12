"""Streamlit dashboard — runtime entrypoint is `app.py`.

Run:
    uv run streamlit run src/panoptes/dashboard/app.py -- --db runs/v1.duckdb

The dashboard reads duckdb directly via `st.cache_data` so a 1k-row file
loads in well under 2 seconds on a warm cache.
"""
