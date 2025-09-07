#!/bin/bash
# Huey One-Click Launcher for macOS
# Double-click this file to start Huey

echo "🧠 Starting Huey..."
echo "================================"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "huey_env" ]; then
    echo "❌ Virtual environment not found. Running installer first..."
    python3 install.py
    if [ $? -ne 0 ]; then
        echo "❌ Installation failed. Please contact support."
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

# Activate virtual environment
echo "🔄 Activating Huey environment..."
source huey_env/bin/activate

# Check all dependencies before starting
echo "🔍 Checking dependencies..."
python3 check_huey_deps.py
if [ $? -ne 0 ]; then
    echo "❌ Dependencies check failed. Cannot start Huey."
    read -p "Press Enter to exit..."
    exit 1
fi

# Start Huey
echo "🚀 Starting Huey web interface..."
echo "🌐 Huey will open in your browser automatically"
echo "🛑 To stop Huey: Close this window or press Ctrl+C"
echo "================================"

# Start Streamlit
python3 -m streamlit run huey_web_interface.py --server.port=8501 --server.address=localhost

echo ""
echo "🛑 Huey stopped"
read -p "Press Enter to close..."