# 💻 Huey Alpha - System Requirements

## 🎯 **Minimum System Requirements**

### **Operating System**
- **Windows:** Windows 10 (64-bit) or newer
- **macOS:** macOS 10.14 (Mojave) or newer
- **Linux:** Ubuntu 18.04, CentOS 7, or equivalent distributions

### **Hardware Requirements**
- **RAM:** 4GB minimum (8GB strongly recommended)
- **Storage:** 2GB free disk space for installation
- **CPU:** Any modern processor (multi-core recommended)
- **Network:** Internet connection for initial setup only

### **Software Dependencies**
- **Python:** 3.8 or higher (3.9-3.11 recommended)
- **Browser:** Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Display:** 1024x768 minimum resolution

## ⚡ **Recommended Configuration**

### **Optimal Performance**
- **RAM:** 8GB+ (16GB for large conversation analysis)
- **Storage:** SSD for better I/O performance
- **CPU:** Quad-core or better (faster eigenvalue computation)
- **Browser:** Chrome or Firefox (best Streamlit compatibility)
- **Display:** 1920x1080 or higher (better visualization)

### **For Large-Scale Testing**
- **RAM:** 16GB+ recommended
- **Storage:** 10GB+ free space (for result caching)
- **CPU:** 8+ cores (parallel processing benefits)
- **Network:** Stable connection (package updates)

## 📊 **Performance Expectations by System**

### **Minimum System (4GB RAM, Dual-core)**
- **Small conversations** (1-2 pages): 30 seconds - 2 minutes
- **Medium conversations** (5-10 pages): 2-5 minutes
- **Large conversations** (20+ pages): 5-15 minutes
- **Maximum practical size:** ~30 pages

### **Recommended System (8GB RAM, Quad-core)**
- **Small conversations:** 10-30 seconds
- **Medium conversations:** 30 seconds - 2 minutes
- **Large conversations:** 2-8 minutes
- **Very large conversations** (50+ pages): 5-20 minutes

### **High-End System (16GB+ RAM, 8+ cores)**
- **Small conversations:** 5-15 seconds
- **Medium conversations:** 15-60 seconds
- **Large conversations:** 1-5 minutes
- **Very large conversations:** 2-10 minutes
- **Extreme conversations** (100+ pages): 5-30 minutes

## 🔍 **Detailed Compatibility**

### **Python Version Compatibility**
| Python Version | Support Status | Notes |
|---------------|----------------|--------|
| 3.7 and below | ❌ Not Supported | Missing required features |
| 3.8 | ✅ Supported | Minimum version |
| 3.9 | ✅ Fully Supported | Recommended |
| 3.10 | ✅ Fully Supported | Recommended |
| 3.11 | ✅ Fully Supported | Best performance |
| 3.12+ | ⚠️ Untested | May work but not verified |

### **Operating System Specific Notes**

#### **Windows**
- **Windows 11:** Fully supported, best performance
- **Windows 10:** Fully supported (version 1909+)
- **Windows 8.1:** May work but not officially supported
- **32-bit Windows:** Not recommended (memory limitations)

**Windows-Specific Requirements:**
- Visual Studio Build Tools (for some packages)
- PowerShell or Command Prompt access
- Administrator rights (for installation)

#### **macOS**
- **Apple Silicon (M1/M2):** Fully supported with native Python
- **Intel Macs:** Fully supported
- **macOS 12+:** Optimal performance
- **macOS 10.14-11:** Supported with possible minor issues

**macOS-Specific Requirements:**
- Xcode Command Line Tools (`xcode-select --install`)
- Homebrew recommended for package management
- Terminal access

#### **Linux**
- **Ubuntu:** 18.04+ (20.04+ recommended)
- **CentOS/RHEL:** 7+ (8+ recommended)
- **Fedora:** 32+ supported
- **Debian:** 10+ supported
- **Arch Linux:** Current versions supported

**Linux-Specific Requirements:**
- Development packages (`build-essential`, `python3-dev`)
- Package manager access (`apt`, `yum`, `pacman`)
- Terminal/shell access

## 🌐 **Browser Compatibility**

### **Fully Supported**
- **Google Chrome:** 90+ (recommended)
- **Mozilla Firefox:** 88+ (recommended)
- **Microsoft Edge:** 90+ (Chromium-based)
- **Safari:** 14+ (macOS only)

### **Limited Support**
- **Internet Explorer:** Not supported
- **Older browser versions:** May have display issues
- **Mobile browsers:** Not optimized for mobile use

### **Browser-Specific Features**
- **File upload:** All supported browsers
- **3D visualizations:** Chrome/Firefox optimal
- **PDF export:** Chrome recommended
- **Keyboard shortcuts:** Chrome/Firefox best support

## 📈 **Memory Usage Patterns**

### **Base Installation**
- **Python environment:** ~200MB
- **Huey idle:** ~150MB
- **Streamlit server:** ~100MB
- **Browser tab:** ~50-200MB

### **During Analysis**
| Conversation Size | Peak Memory Usage |
|------------------|------------------|
| Small (1-2 pages) | ~500MB |
| Medium (5-10 pages) | ~800MB - 1.5GB |
| Large (20+ pages) | ~1.5GB - 4GB |
| Very Large (50+ pages) | ~3GB - 8GB |

### **Memory Management**
- Memory usage spikes during eigenvalue computation
- Returns to baseline after analysis complete
- Multiple concurrent analyses multiply memory usage
- Browser caching adds ~100MB per session

## ⚠️ **Performance Warning Signs**

### **System Struggling Indicators**
- Processing takes >3x expected time
- System becomes unresponsive during analysis
- Memory usage grows continuously without plateau
- Fan noise increases dramatically
- Other applications become sluggish

### **When to Stop Testing**
- System temperature warnings appear
- Memory usage exceeds 90% of available RAM
- Processing hasn't completed after 1 hour
- System stability is compromised

## 🧪 **Testing Recommendations by System**

### **Minimum Systems (4GB RAM)**
- Start with provided small test files only
- Monitor memory usage closely
- Test one analysis at a time
- Expect longer processing times
- Consider system upgrades if possible

### **Recommended Systems (8GB+ RAM)**
- Can handle all provided test files
- Try medium to large conversations
- Multiple concurrent tests possible
- Good for comprehensive stability testing
- Suitable for full alpha testing protocol

### **High-End Systems (16GB+ RAM)**
- Ideal for stress testing
- Can handle very large conversations
- Perfect for performance benchmarking
- Best for extensive stability testing
- Can run multiple Huey instances

## 🔧 **Optimization Tips**

### **Before Installation**
- Close unnecessary applications
- Ensure adequate free disk space
- Update Python to latest stable version
- Install on SSD if available

### **During Installation**
- Use virtual environment (strongly recommended)
- Install during off-peak hours (fewer interruptions)
- Have stable internet connection
- Monitor installation progress

### **During Testing**
- Start with small files, work up to larger ones
- Monitor system resources during processing
- Take breaks between large analyses
- Save results frequently
- Document performance observations

### **Troubleshooting Performance**
- Restart Streamlit between large analyses
- Clear browser cache periodically
- Monitor system temperature
- Close other resource-intensive applications
- Consider processing during cooler parts of day

---

## 📞 **Hardware Upgrade Recommendations**

If your system struggles with Huey Alpha:

### **Priority 1: More RAM**
- 8GB minimum for comfortable use
- 16GB for large conversation analysis
- RAM is the biggest performance factor

### **Priority 2: SSD Storage**
- Significant improvement in file I/O
- Faster Python package loading
- Better overall system responsiveness

### **Priority 3: Better CPU**
- Multi-core processors help with eigenvalue computation
- Modern CPUs handle matrix operations better
- Hyperthreading provides modest benefits

Remember: Huey is computationally intensive by design - it's doing sophisticated mathematical analysis on high-dimensional data! 🧠💪