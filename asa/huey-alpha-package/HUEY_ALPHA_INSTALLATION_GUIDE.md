# 🧠 Huey Alpha - Installation Guide
## Remote Installation for Galileo Company Team

**Version:** Alpha 0.1  
**Target:** Galileo Company team members at remote sites  
**Platform:** Windows, macOS, Linux

---

## 📦 **What's Included in This Package**

```
huey-alpha-package/
├── HUEY_ALPHA_INSTALLATION_GUIDE.md (this file)
├── HUEY_ALPHA_TESTER_GUIDE.md (testing protocols)
├── alpha_testing_checklist.md (quick reference)
├── requirements.txt (Python dependencies)
├── install.py (automated installer)
├── huey_complete_platform.py (core engine)
├── huey_speaker_detector.py (conversation parsing)
├── huey_web_interface.py (web interface)
├── conversational_self_concept_experiment.py (analysis tools)
├── experimental_network_complete.py (network engine)
├── test_data/
│   ├── chat_transcript_fixed.txt
│   ├── sample_conversation.txt
│   └── README_test_data.md
└── troubleshooting/
    ├── common_issues.md
    └── system_requirements.md
```

---

## 🚀 **Quick Start (5 Minutes)**

### **Step 1: Check Python**
Open terminal/command prompt and run:
```bash
python --version
```
**Need:** Python 3.8 or higher. If not installed, get it from [python.org](https://python.org)

### **Step 2: Extract Package**
- Unzip `huey-alpha-package.zip` to your preferred location
- Navigate to the folder in terminal:
```bash
cd path/to/huey-alpha-package
```

### **Step 3: Auto-Install**
Run the automated installer:
```bash
python install.py
```
This will:
- Create virtual environment
- Install all dependencies
- Test the installation
- Launch Huey automatically

### **Step 4: Access Huey**
- Browser will open automatically to `http://localhost:8501`
- If not, manually navigate to that address
- You should see the Huey interface with "Powered by The Galileo Company"

### **Step 5: Quick Test**
- Go to the "Upload" tab
- Select `test_data/chat_transcript_fixed.txt`
- Click "Process Text"
- Verify you get results without errors

**✅ Installation Complete!**

---

## 🔧 **Manual Installation (If Auto-Install Fails)**

### **Prerequisites**
- Python 3.8+ with pip
- 4GB+ RAM recommended
- Modern web browser
- 2GB free disk space

### **Step-by-Step Manual Setup**

#### **1. Create Virtual Environment**
```bash
# Windows
python -m venv huey_env
huey_env\Scripts\activate

# macOS/Linux  
python3 -m venv huey_env
source huey_env/bin/activate
```

#### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

#### **3. Test Installation**
```bash
python -c "import streamlit, numpy, pandas, plotly; print('Dependencies OK')"
```

#### **4. Launch Huey**
```bash
streamlit run huey_web_interface.py --server.port=8501
```

#### **5. Open Browser**
Navigate to: `http://localhost:8501`

---

## 🧪 **Verify Installation**

### **Smoke Test Checklist**
- [ ] Huey web interface loads without errors
- [ ] "Powered by The Galileo Company" appears under title
- [ ] Can upload/paste text in the interface
- [ ] Processing test file produces results
- [ ] 3D visualization tab shows dimensional reduction warning
- [ ] Network statistics display properly
- [ ] No Python error messages in terminal

### **Quick Functionality Test**
1. Upload `test_data/chat_transcript_fixed.txt`
2. Process with default settings
3. Check that you get:
   - Network statistics (neurons, connections, mass)
   - Concept associations results
   - 3D visualization with warning
   - Eigenvalue analysis results
4. Verify processing completes in <2 minutes

---

## 🆘 **Troubleshooting**

### **Common Issues & Solutions**

#### **"Python not found"**
```bash
# Try these alternatives:
python3 --version
py --version
```
Install Python from [python.org](https://python.org) if needed.

#### **"Permission denied" errors**
```bash
# Windows: Run as administrator
# macOS/Linux: 
sudo python install.py
```

#### **Streamlit won't start**
```bash
# Kill any existing processes
pkill -f streamlit

# Try alternate port
streamlit run huey_web_interface.py --server.port=8502
```

#### **Import errors during installation**
```bash
# Upgrade pip first
pip install --upgrade pip

# Clean install
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

#### **Browser doesn't open automatically**
- Manually go to `http://localhost:8501`
- Try different browser (Chrome, Firefox, Safari)
- Check if port 8501 is blocked by firewall

#### **"Module not found" at runtime**
```bash
# Ensure virtual environment is activated
# Look for (huey_env) in your prompt

# Windows
huey_env\Scripts\activate

# macOS/Linux
source huey_env/bin/activate
```

### **Getting Help**
1. Check `troubleshooting/common_issues.md` for detailed solutions
2. Contact development team with error details
3. Include system info (OS, Python version) in help requests

---

## 💻 **System Requirements**

### **Minimum Requirements**
- **OS:** Windows 10, macOS 10.14, Ubuntu 18.04 (or equivalent Linux)
- **RAM:** 4GB (8GB recommended for large conversations)
- **Storage:** 2GB free space
- **Python:** 3.8 or higher
- **Browser:** Chrome 90+, Firefox 88+, Safari 14+, Edge 90+

### **Optimal Performance**
- **RAM:** 8GB+ for processing long conversations
- **CPU:** Multi-core processor for faster eigenvalue computation
- **SSD:** For better file I/O performance
- **Network:** Stable connection (for package installation only)

---

## 🔒 **Security & Confidentiality**

### **Data Privacy**
- **All processing is local** - no data sent to external servers
- **No internet required** after installation
- **Your conversations remain on your computer**
- **No telemetry or usage tracking**

### **File Handling**
- Test files provided are anonymized examples
- **Do not share results** outside Galileo Company during alpha phase
- **Anonymize personal data** before processing
- **Delete sensitive results** when testing complete

---

## 📈 **Performance Expectations**

### **Typical Processing Times**
- **Small conversation** (1-2 pages): 10-30 seconds
- **Medium conversation** (5-10 pages): 1-3 minutes  
- **Large conversation** (20+ pages): 3-10 minutes
- **Very large** (50+ pages): 10-30 minutes

### **Memory Usage**
- **Base system**: ~200MB
- **Small text**: +100MB
- **Large text**: +500MB-2GB
- **Peak during analysis**: 2x normal usage

### **When to Worry**
- Processing takes >10x expected time
- Memory usage grows continuously without stopping
- System becomes unresponsive
- Temperature/fan noise increases dramatically

---

## 🔄 **Updates & Maintenance**

### **During Alpha Period**
- **No automatic updates** - package is self-contained
- **New versions** will be distributed separately
- **Backup your results** before installing updates
- **Document version** in all test reports

### **Keeping Things Clean**
```bash
# Deactivate virtual environment when done
deactivate

# To completely remove installation:
rm -rf huey-alpha-package  # macOS/Linux
rmdir /s huey-alpha-package  # Windows
```

---

## 🚨 **Important Reminders**

### **Alpha Software Warnings**
- This is **experimental software** - expect bugs
- **Save results frequently** - system may crash
- **Not for production use** - alpha testing only
- **Report all issues** to development team

### **Scientific Limitations**
- Read the **DIMENSIONAL REDUCTION WARNING** in 3D visualizations
- Only ~15-30% of network structure visible in plots
- Use full eigenvalue analysis for quantitative work
- Results need validation through testing protocols

### **Confidentiality**
- **Galileo Company internal use only**
- **No external sharing** of software or results
- **Anonymize test data** before processing
- **Report findings** only within team channels

---

## 📞 **Support Contacts**

**Installation Issues:** Development Team  
**Scientific Questions:** Joseph Woelfel  
**Bug Reports:** Alpha Testing Coordinator  
**Testing Protocol Help:** Reference HUEY_ALPHA_TESTER_GUIDE.md

---

## ✅ **Installation Success Checklist**

- [ ] Python 3.8+ confirmed working
- [ ] Virtual environment created and activated
- [ ] All dependencies installed without errors
- [ ] Huey launches and interface loads
- [ ] "Powered by The Galileo Company" visible
- [ ] Test file processes successfully
- [ ] 3D visualization shows proper warnings
- [ ] No error messages during smoke test
- [ ] Ready to begin alpha testing protocols

**🎉 Welcome to Huey Alpha Testing!**

Proceed to `HUEY_ALPHA_TESTER_GUIDE.md` for detailed testing instructions.

---

**Package Version:** Alpha 0.1  
**Installation Guide Version:** 1.0  
**Last Updated:** August 25, 2025