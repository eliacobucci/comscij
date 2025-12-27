#!/bin/bash
# Permanent Huey Startup Script with ARM Python and JAX Metal
# This ensures JAX always works correctly on Apple Silicon

cd /Users/josephwoelfel/asa

# Check if ARM environment exists
if [ ! -d "huey_arm_env" ]; then
    echo "❌ ARM environment not found. Creating it..."
    /opt/homebrew/bin/python3.13 -m venv huey_arm_env
    source huey_arm_env/bin/activate
    pip install -r requirements_huey_arm.txt
    echo "✅ ARM environment created"
else
    echo "✅ ARM environment found"
fi

# Activate environment
source huey_arm_env/bin/activate

# Verify JAX version
JAX_VERSION=$(python -c "import jax; print(jax.__version__)" 2>/dev/null)
if [ "$JAX_VERSION" != "0.4.34" ]; then
    echo "⚠️  JAX version is $JAX_VERSION, reinstalling 0.4.34..."
    pip install -r requirements_huey_arm.txt
fi

echo "🧠 Starting Huey with JAX Metal on Apple Silicon..."
echo "🌐 Will open at: http://localhost:8502"
echo ""

streamlit run huey_time_working.py
