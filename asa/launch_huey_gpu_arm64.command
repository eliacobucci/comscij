#!/bin/bash
# Launch Huey GPU Web Interface with ARM64 Python for JAX Metal acceleration

echo "🚀 Launching HueyGPU with ARM64 Python for JAX Metal acceleration..."
echo "=================================================================="

# Change to the Huey directory
cd "$(dirname "$0")"

# Force ARM64 architecture and use universal Python
echo "   🏗️  Forcing ARM64 architecture for JAX Metal GPU support..."
arch -arm64 /usr/local/bin/python3 -c "
import platform
print(f'   ✅ Architecture: {platform.machine()}')
print(f'   ✅ Platform: {platform.platform()}')

try:
    import jax
    print(f'   ✅ JAX devices: {jax.devices()}')
    if 'metal' in str(jax.devices()):
        print('   🚀 JAX Metal GPU acceleration ENABLED!')
    else:
        print('   ⚠️  JAX Metal not detected')
except ImportError:
    print('   ❌ JAX not installed')
"

echo ""
echo "   🌐 Starting Streamlit on localhost:8505..."
echo "   📱 Open your browser to: http://localhost:8505"
echo "   🛑 Press Ctrl+C to stop"
echo ""

# Launch with ARM64 Python
exec arch -arm64 /usr/local/bin/python3 -m streamlit run huey_gpu_web_interface_complete.py --server.port=8505 --server.address=localhost