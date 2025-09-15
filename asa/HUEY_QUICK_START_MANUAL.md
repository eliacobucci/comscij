# 🧠 Huey AI Conversational Analysis
## Professional Quick Start Guide

**Version:** 4.0  
**Date:** September 2025  
**Developer:** The Galileo Company  

---

## 📋 Overview

**Huey** is an advanced AI conversational analysis platform that transforms dialogue into interactive 3D neural network visualizations. Using cutting-edge Hebbian learning algorithms, Huey reveals hidden patterns in conversations and maps speaker relationships in multidimensional space.

---

## 🚀 Quick Start Guide

### Step 1: Launch Huey
```bash
streamlit run huey_gpu_web_interface_complete.py --server.port=8505
```

### Step 2: Access Web Interface
Open your browser and navigate to:
```
http://localhost:8505
```

### Step 3: Upload Conversation File
1. Click **"Choose file"** button
2. Select your conversation text file
3. Huey automatically detects speakers and processes exchanges
4. Wait for neural network processing to complete

### Step 4: Explore Results
- 📊 **Network Statistics** - View neurons, connections, processing metrics
- 🌐 **3D Visualization** - Interactive exploration of concept relationships  
- ⚡ **Performance Analytics** - GPU vs CPU acceleration insights

---

## 📄 File Format Requirements

Huey accepts plain text files with natural conversation format:

```
Speaker A: Hello, how are you today?
Speaker B: I'm doing well, thank you for asking.
Speaker A: That's wonderful to hear!
Speaker B: How has your day been going?
```

**Supported formats:** `.txt`, `.dat`, `.log`

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **Automatic Speaker Detection** | Intelligent identification of conversation participants |
| 🚀 **GPU Acceleration** | Optimal processing with JAX Metal on Apple Silicon |
| 📈 **Real-time Visualization** | Interactive 3D neural network mapping |
| 🌍 **Multi-language Support** | Analysis in English, German, Mandarin, Hindi, and more |
| 🔬 **Advanced Analytics** | Concept mass, directional semantics, speaker patterns |

---

## ⚡ Performance Optimization

### Automatic Acceleration Selection
- **Small files** (< 25 exchanges): CPU processing optimized
- **Large files** (≥ 25 exchanges): GPU acceleration recommended
- **Apple Silicon**: Use ARM64 Python for maximum performance

### Performance Benchmarks
| File Size | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| 10 exchanges | 0.02s | 0.03s | 0.7x |
| 25 exchanges | 0.05s | 0.05s | 1.0x |
| 100 exchanges | 0.20s | 0.06s | 3.4x |
| 1000 exchanges | 2.1s | 0.13s | 16.4x |

---

## 🔧 Troubleshooting

### Common Issues

**🐌 Slow Processing**
- Ensure ARM64 Python for GPU acceleration
- Check available system memory
- Verify JAX Metal installation

**🌐 Connection Errors** 
- Restart Streamlit server
- Check port 8505 availability
- Clear browser cache

**📊 Empty Visualization**
- Verify file contains actual dialogue
- Check speaker detection threshold
- Ensure minimum exchange count (≥2)

**💾 Memory Issues**
- Reduce file size or use chunking
- Close other applications
- Restart Python session

---

## 📞 Support & Resources

**Technical Support:** support@galileocompany.com  
**Documentation:** galileocompany.com/docs  
**GitHub:** github.com/galileocompany/huey  

---

## 📚 Citation

When using Huey in research, please cite:
> Huey AI Conversational Analysis Platform (Version 4.0). (2025). The Galileo Company.

---

**© 2025 The Galileo Company. All rights reserved.**  
*Advanced Neural Network Analysis • Multidimensional Scaling • Conversational Intelligence*