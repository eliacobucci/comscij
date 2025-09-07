@echo off
REM Huey One-Click Launcher for Windows
REM Double-click this file to start Huey

echo 🧠 Starting Huey...
echo ================================

REM Get the directory where this script is located
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "huey_env" (
    echo ❌ Virtual environment not found. Running installer first...
    python install.py
    if errorlevel 1 (
        echo ❌ Installation failed. Please contact support.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo 🔄 Activating Huey environment...
call huey_env\Scripts\activate.bat

REM Start Huey
echo 🚀 Starting Huey web interface...
echo 🌐 Huey will open in your browser automatically
echo 🛑 To stop Huey: Close this window or press Ctrl+C
echo ================================

REM Start Streamlit
python -m streamlit run huey_web_interface.py --server.port=8501 --server.address=localhost

echo.
echo 🛑 Huey stopped
pause