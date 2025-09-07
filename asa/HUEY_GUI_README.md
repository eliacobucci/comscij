# 🚀 Huey GUI GPU - Professional Tkinter Interface

A professional Tkinter-based GUI for the Huey GPU Hebbian Self-Concept Analysis Platform with full Galileo branding support.

## ✨ Features

### 🎯 Core Functionality
- **GPU-Accelerated Processing** - Revolutionary JAX/NumPy acceleration targeting O(n²) bottlenecks
- **Conversation Mode Toggle** - Handle both dialogues and single-author texts (Wikipedia, articles, etc.)
- **File Processing** - Support for TXT and PDF files with automatic speaker detection
- **Real-time Progress** - Live processing updates with progress bars and ETA
- **3D Visualization** - Interactive matplotlib integration for concept space exploration

### 🎨 Professional Interface
- **Tabbed Interface** - Organized workflow with File Processing, Results, Visualization, and Settings
- **Galileo Branding** - Full branding support with custom logos, colors, and styling
- **Responsive Design** - Professional layout that scales with window resizing
- **Status Indicators** - Visual feedback for GPU status and processing states

### 📊 Analysis Tools
- **Export Options** - JSON, CSV export for further analysis
- **Detailed Results** - Comprehensive analysis summaries with metrics
- **Performance Monitoring** - GPU acceleration statistics and timing
- **Speaker Analysis** - Multi-speaker conversation processing

## 🚀 Quick Start

### 1. Launch the Application
```bash
# Simple launch
python3 huey_gui_gpu.py

# Or use the launcher with dependency checking
python3 launch_huey_gui.py
```

### 2. Add Your Galileo Branding (Optional)
Place these files in the same directory:
- `galileo_logo.png` (recommended: 200x100px)
- `galileo_banner.png` (recommended: 1200x150px) 
- `galileo_icon.ico` (recommended: 32x32px)

**Note:** The app works perfectly without these files - it creates professional branded placeholders automatically!

### 3. Process Your First File
1. Click **"Browse..."** to select a conversation file (TXT or PDF)
2. Toggle **"🗨️ Conversation Mode"** based on your content:
   - ✅ **ON** for dialogues and conversations
   - ❌ **OFF** for Wikipedia articles, single-author texts
3. Click **"🚀 Process File"** to start analysis
4. Watch real-time progress and view results in the **Results** tab

## 🔧 Configuration

### Network Settings
- **Max Neurons**: 50-500 (default: 100)
- **Timeout**: Processing time limit in hours
- **Exchange Limit**: Maximum exchanges to process
- **GPU Acceleration**: Toggle between GPU and CPU modes

### Advanced Options
- **Export Results**: JSON and CSV export functionality
- **Visualization**: 3D concept space exploration
- **Performance Stats**: GPU acceleration metrics

## 📁 File Structure

```
asa/
├── huey_gui_gpu.py              # Main GUI application
├── huey_gui_branding.py         # Branding and image management
├── launch_huey_gui.py           # Professional launcher script
├── huey_gpu_conversational_experiment.py  # Core GPU engine
├── huey_speaker_detector.py     # Speaker detection system
└── [Optional Branding Files]
    ├── galileo_logo.png
    ├── galileo_banner.png
    └── galileo_icon.ico
```

## 🎨 Branding Customization

### Color Scheme
The interface uses a professional Galileo color palette:
- **Primary Blue**: #2E4057 (Deep, professional)
- **Teal Accent**: #048A81 (Modern, fresh) 
- **Orange Highlight**: #F39C12 (Attention-grabbing)
- **Success Green**: #27AE60 (Positive feedback)
- **Warning Orange**: #E67E22 (Caution)
- **Light Background**: #ECF0F1 (Clean, minimal)

### Logo Integration
```python
# The branding manager automatically handles:
logo_img = branding.load_image('galileo_logo.png', (120, 80))

# Falls back to professional placeholder if file not found
if not logo_img:
    logo_img = branding.create_placeholder_logo((120, 80))
```

## 🚀 Comparison with Streamlit Version

### Advantages of Tkinter GUI
✅ **Native Desktop Experience** - No browser required  
✅ **Custom Branding** - Full control over appearance  
✅ **Better Performance** - Direct system integration  
✅ **Professional Look** - Desktop application aesthetics  
✅ **Offline Operation** - No web server needed  
✅ **File System Integration** - Native file dialogs  

### Feature Parity
- ✅ Full GPU acceleration support
- ✅ Conversation mode toggle  
- ✅ File processing (TXT/PDF)
- ✅ Real-time progress monitoring
- ✅ Results visualization
- ✅ Export functionality
- ✅ Performance metrics

## 🔧 Dependencies

### Required
- `tkinter` (usually included with Python)
- `numpy`
- `matplotlib` 
- `pandas`
- `Pillow` (PIL)

### Install Dependencies
```bash
pip install numpy matplotlib pandas pillow
```

## 🌟 Usage Tips

1. **For Wikipedia/Articles**: Uncheck "Conversation Mode" to treat as single-author
2. **For Dialogues**: Keep "Conversation Mode" checked for speaker detection
3. **Large Files**: Increase timeout and exchange limit in settings
4. **Custom Branding**: Drop your logo files in the directory - they're automatically detected
5. **Performance**: Enable GPU acceleration for files with 100+ exchanges

## 📊 Example Workflow

1. **Load File**: Browse and select `Richard_Feynman.pdf`
2. **Configure**: Disable conversation mode (single author)
3. **Process**: Click "Process File", watch progress
4. **Analyze**: Review results in Results tab
5. **Visualize**: Explore 3D concept space in Visualization tab
6. **Export**: Save results as JSON or CSV for further analysis

## 🎯 Perfect For

- **Research Projects** - Academic analysis of conversational data
- **Professional Analysis** - Corporate communication studies  
- **Educational Use** - Teaching Hebbian learning concepts
- **Custom Branding** - Organizations wanting branded analysis tools
- **Desktop Integration** - Users preferring native applications

---

**Created with ❤️ for the Galileo Research Framework**  
*Professional Hebbian Self-Concept Analysis Platform*