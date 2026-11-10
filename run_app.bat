@echo off
cd /d "%~dp0"
if exist "env\Scripts\streamlit.exe" (
    echo Launching UV-Vis Spectral Analyzer using local virtual environment...
    "env\Scripts\streamlit.exe" run streamlit_app.py
) else (
    echo Launching UV-Vis Spectral Analyzer using system Python...
    python -m streamlit run streamlit_app.py
)
pause
