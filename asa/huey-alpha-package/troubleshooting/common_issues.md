# 🆘 Huey Alpha - Common Issues & Solutions

## 🐛 **Installation Issues**

### **Python Version Problems**
```
Error: "Python 3.x required, found Python 2.7"
```
**Solution:**
- Install Python 3.8+ from [python.org](https://python.org)
- Try `python3` instead of `python`
- On Windows, use `py -3` command

### **Virtual Environment Creation Failed**
```
Error: "No module named venv"
```
**Solution:**
```bash
# Install venv module
sudo apt-get install python3-venv  # Ubuntu/Debian
brew install python  # macOS with Homebrew

# Alternative: Use virtualenv
pip install virtualenv
virtualenv huey_env
```

### **Permission Denied Errors**
```
Error: "Permission denied" during installation
```
**Solution:**
```bash
# Windows: Run as Administrator
# Right-click Command Prompt → "Run as Administrator"

# macOS/Linux: Use sudo carefully
sudo python install.py
# OR fix permissions:
chmod +x install.py
```

### **Pip Install Failures**
```
Error: "Failed to build wheels for [package]"
```
**Solutions:**
```bash
# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install build tools (if needed):
# Windows: Install Visual Studio Build Tools
# macOS: xcode-select --install
# Linux: sudo apt-get install build-essential python3-dev

# Try individual package installation:
pip install streamlit
pip install pandas
# etc...
```

## 🌐 **Network & Firewall Issues**

### **Port 8501 Already in Use**
```
Error: "Address already in use"
```
**Solutions:**
```bash
# Find and kill existing process
netstat -tulpn | grep :8501
kill -9 [PID]

# Or use different port
streamlit run huey_web_interface.py --server.port=8502
```

### **Browser Won't Open**
```
Issue: Streamlit starts but browser doesn't open
```
**Solutions:**
- Manually navigate to `http://localhost:8501`
- Try different browser (Chrome, Firefox, Safari)
- Check firewall settings
- Disable VPN temporarily
- Use incognito/private browsing mode

### **Firewall Blocking Access**
```
Issue: Can't access localhost:8501
```
**Solutions:**
- Add firewall exception for Python/Streamlit
- Temporarily disable firewall for testing
- Use `--server.address=127.0.0.1` flag
- Try `http://127.0.0.1:8501` instead of `localhost`

## 💾 **Runtime Issues**

### **Memory Errors During Processing**
```
Error: "MemoryError" or system becomes unresponsive
```
**Solutions:**
- Use smaller text files for testing
- Reduce max_neurons parameter (try 200-300)
- Close other applications
- Restart Python/Streamlit
- Consider 64-bit Python if using 32-bit

### **Slow Processing Times**
```
Issue: Analysis takes >10 minutes for small files
```
**Diagnostic Steps:**
1. Check CPU usage (should be high during processing)
2. Check memory usage (shouldn't grow continuously)
3. Test with minimal text first
4. Restart Streamlit session

**Solutions:**
- Reduce window_size parameter (try 5)
- Lower learning_rate (try 0.1)
- Use SSD instead of HDD if possible
- Close background applications

### **Import Errors at Runtime**
```
Error: "ModuleNotFoundError: No module named 'streamlit'"
```
**Solutions:**
```bash
# Ensure virtual environment is activated
# Windows:
huey_env\Scripts\activate

# macOS/Linux:
source huey_env/bin/activate

# Verify activation (should show huey_env)
which python
```

## 📊 **Analysis Issues**

### **No Results Generated**
```
Issue: Processing completes but no results shown
```
**Diagnostic Steps:**
1. Check if file uploaded correctly
2. Look for error messages in terminal
3. Try with provided test files
4. Check file format (should be plain text)

**Solutions:**
- Ensure text contains recognizable conversation
- Use UTF-8 encoding
- Remove special characters
- Try smaller file first

### **Inconsistent Results**
```
Issue: Same file produces different results each run
```
**Expected vs Concerning:**
- **Small variations:** Normal (random seed effects)
- **Major differences:** Concerning (stability issue)

**Solutions:**
- Document exact differences observed
- Test with multiple file sizes
- Report to development team
- Note system specifications

### **Dimensional Reduction Warning**
```
Issue: 3D plot shows low variance (< 10%)
```
**This is NORMAL:**
- 3D plots are intentionally limited
- Real analysis happens in 250+ dimensions
- Warning is there to prevent misinterpretation
- Use eigenvalue analysis for quantitative work

## 🔄 **System Compatibility**

### **macOS Issues**
```
Common: "Apple Silicon" compatibility
```
**Solutions:**
- Use native Python for Apple Silicon
- Try Rosetta mode if needed: `arch -x86_64 python install.py`
- Install Xcode command line tools: `xcode-select --install`

### **Windows Issues**
```
Common: Path length limitations, encoding issues
```
**Solutions:**
- Enable long path support in Windows 10+
- Use PowerShell instead of Command Prompt
- Ensure UTF-8 encoding: `chcp 65001`
- Run as Administrator if permission issues

### **Linux Issues**
```
Common: Missing development packages
```
**Solutions:**
```bash
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install python3-dev python3-pip build-essential

# CentOS/RHEL:
sudo yum install python3-devel python3-pip gcc

# Arch:
sudo pacman -S python python-pip base-devel
```

## 🧪 **Testing Issues**

### **Stability Tests Failing**
```
Issue: Results vary dramatically between runs
```
**Investigation:**
1. Document exact differences
2. Check system resource usage
3. Try with different text files
4. Note random seed sensitivity

**Report to team:**
- System specifications
- File size/type used
- Magnitude of differences
- Reproducibility across restarts

### **Function Word Contamination**
```
Issue: Top concepts are "the", "and", "is", etc.
```
**This is a KNOWN ISSUE:**
- Document which function words appear
- Note their relative weights
- Report to development team
- Still proceed with testing other features

## 📞 **Getting Help**

### **Information to Include**
When reporting issues, please include:
- **Operating System:** (Windows 10, macOS 12.1, Ubuntu 20.04, etc.)
- **Python Version:** `python --version`
- **Error Messages:** Complete error text
- **Steps to Reproduce:** Exact sequence that caused issue
- **File Information:** Size, type, content description
- **System Resources:** RAM, CPU, available disk space

### **Escalation Criteria**
**Report immediately for:**
- Data corruption or loss
- System crashes requiring restart
- Security or privacy concerns
- Results that contradict established theory

**Weekly reports for:**
- Performance issues
- Usability problems
- Feature requests
- Stability concerns

### **Contact Methods**
- **Critical Issues:** Direct contact to development team
- **General Issues:** Document in weekly testing report
- **Questions:** Reference main testing guide first

---

## 🎯 **Success Metrics**

### **Installation Success:**
- ✅ All dependencies install without errors
- ✅ Huey launches and web interface loads
- ✅ Test file processes successfully
- ✅ No crashes during smoke test

### **Operational Success:**
- ✅ Consistent results across multiple runs
- ✅ Processing times within expected ranges
- ✅ Memory usage stable and reasonable
- ✅ All analysis tabs function properly

### **Scientific Success:**
- ✅ Eigenvalue patterns make theoretical sense
- ✅ Concept associations are meaningful
- ✅ Self-concept clusters align with expectations
- ✅ Results provide novel insights

Remember: Finding issues during alpha testing is SUCCESS, not failure. Every bug caught now prevents problems later! 🐛→✅