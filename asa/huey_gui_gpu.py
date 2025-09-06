#!/usr/bin/env python3
"""
🚀 Huey GUI GPU - Tkinter Interface
==================================

A professional Tkinter-based GUI for the Huey GPU Hebbian Self-Concept Analysis Platform.
Designed for Galileo branding and custom visual elements.

Copyright (c) 2025 Emary Iacobucci and Joseph Woelfel. All rights reserved.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from matplotlib import cm
import matplotlib.patheffects as path_effects
import threading
import queue
import time
import json
import tempfile
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import Huey components
try:
    from huey_gpu_conversational_experiment import HueyGPUConversationalNetwork
    from huey_speaker_detector import HueySpeakerDetector
    from huey_gui_branding import HueyBrandingManager
except ImportError as e:
    print(f"❌ Could not import Huey components: {e}")
    exit(1)

class HueyGUIGPU:
    """Professional Tkinter GUI for Huey GPU with Galileo branding support."""
    
    def __init__(self):
        """Initialize the Huey GUI GPU application."""
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("🚀 Huey GPU - Hebbian Self-Concept Analysis")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Initialize branding manager
        self.branding = HueyBrandingManager()
        
        # Configure style with branding
        self.setup_styles()
        
        # Initialize Huey network
        self.huey = None
        self.processing_thread = None
        self.results_queue = queue.Queue()
        
        # State variables
        self.conversation_mode = tk.BooleanVar(value=True)
        self.max_neurons = tk.IntVar(value=100)
        self.timeout_hours = tk.DoubleVar(value=2.0)
        self.exchange_limit = tk.IntVar(value=10000)
        self.dark_mode = tk.BooleanVar(value=False)
        self.config_complete = False
        
        # Initialize theme color schemes - Original professional light theme
        self.light_colors = {
            'bg': '#ECF0F1',        # Original Galileo light background
            'fg': '#2C3E50',        # Original Galileo dark text
            'select_bg': '#3498DB', # Original blue selection
            'select_fg': '#FFFFFF', # White selected text
            'insert_bg': '#2C3E50', # Dark cursor
            'entry_bg': '#FFFFFF',  # White entry fields
            'button_bg': '#FFFFFF', # White buttons
            'frame_bg': '#ECF0F1',  # Original light frames
        }
        
        self.dark_colors = {
            'bg': '#2C3E50',        # Dark background
            'fg': '#ECF0F1',        # Light text
            'select_bg': '#3498DB', # Blue selection
            'select_fg': '#FFFFFF', # White selected text
            'insert_bg': '#ECF0F1', # Light cursor
            'entry_bg': '#34495E',  # Dark entry fields
            'button_bg': '#34495E', # Dark buttons
            'frame_bg': '#2C3E50',  # Dark frames
        }
        
        # Initialize text widgets list for theme updates
        self.text_widgets = []
        
        # GLOBAL DATA STATE for tab synchronization
        self.global_data_state = {
            'data_processed': False,
            'processing_complete': False,
            'results_available': False,
            'visualization_ready': False,
            'last_update': None
        }
        
        # Main visualization zoom and pan state
        self.main_zoom_level = 1.0
        self.main_pan_start = None
        self.main_is_panning = False
        self.main_original_xlim = None
        self.main_original_ylim = None
        self.main_original_zlim = None
        
        # Start with configuration screen
        self.create_config_screen()
        
        # Initialize Huey network (will be done after config)
        self.huey = None
        
        # Start periodic state monitoring
        self.monitor_global_state()
        
        print("🚀 Huey GUI GPU initialized successfully!")
    
    def setup_styles(self):
        """Configure professional styling with Galileo branding."""
        
        # Use branding colors
        self.colors = self.branding.brand_colors
        
        # Apply branded window styling
        self.branding.setup_branded_window(self.root)
        
        # Configure ttk styles
        style = ttk.Style()
        
        # Determine if we're in dark mode
        is_dark = self.dark_mode.get() if hasattr(self, 'dark_mode') else False
        
        # Set theme colors based on mode with complete fallback
        if is_dark and hasattr(self, 'dark_colors'):
            theme_colors = self.dark_colors
        elif not is_dark and hasattr(self, 'light_colors'):
            theme_colors = self.light_colors
        else:
            # Complete fallback theme colors (original style)
            theme_colors = {
                'bg': '#ECF0F1', 'fg': '#2C3E50', 'entry_bg': '#FFFFFF', 
                'select_bg': '#3498DB', 'select_fg': '#FFFFFF', 'insert_bg': '#2C3E50',
                'button_bg': '#FFFFFF', 'frame_bg': '#ECF0F1'
            }
        
        # Configure ttk theme
        if is_dark:
            try:
                style.theme_use('clam')  # Use clam theme which supports dark colors better
            except:
                pass
        
        # Header styles
        style.configure('Header.TLabel', 
                       font=('Arial', 16, 'bold'),
                       background=self.colors['galileo_light'],
                       foreground=self.colors['galileo_blue'])
        
        style.configure('Subheader.TLabel',
                       font=('Arial', 12, 'bold'),
                       background=self.colors['galileo_light'],
                       foreground=self.colors['galileo_dark'])
        
        # Frame styles
        style.configure('TFrame',
                       background=self.colors['galileo_light'])
        
        style.configure('TLabelFrame',
                       background=self.colors['galileo_light'],
                       foreground=self.colors['galileo_dark'])
        
        style.configure('TLabelFrame.Label',
                       background=self.colors['galileo_light'],
                       foreground=self.colors['galileo_dark'])
        
        # Notebook styles
        style.configure('TNotebook',
                       background=self.colors['galileo_light'])
        
        style.configure('TNotebook.Tab',
                       background=theme_colors['entry_bg'],
                       foreground=self.colors['galileo_dark'],
                       padding=(10, 5))
        
        style.map('TNotebook.Tab',
                  background=[('selected', self.colors['galileo_light']),
                             ('active', theme_colors['select_bg'])],
                  foreground=[('selected', self.colors['galileo_dark']),
                             ('active', '#FFFFFF')])
        
        # Checkbutton and radiobutton styles
        style.configure('TCheckbutton',
                       background=self.colors['galileo_light'],
                       foreground=self.colors['galileo_dark'])
        
        style.configure('TRadiobutton',
                       background=self.colors['galileo_light'],
                       foreground=self.colors['galileo_dark'])
        
        # Label styles
        style.configure('TLabel',
                       background=self.colors['galileo_light'],
                       foreground=self.colors['galileo_dark'])
        
        # Button styles with high contrast
        self.branding.create_branded_button_style(self.root, 'Primary.TButton')
        
        style.configure('Success.TButton',
                       background='#A9DFBF',  # Light green background
                       foreground=self.colors['galileo_dark'],
                       font=('Arial', 10, 'bold'),
                       borderwidth=2,
                       relief='raised')
        
        style.map('Success.TButton',
                  background=[('active', '#82E0AA'), ('pressed', '#7DCEA0')],  # Lighter green shades
                  foreground=[('active', self.colors['galileo_dark']),
                             ('pressed', self.colors['galileo_dark'])])
        
        style.configure('Warning.TButton', 
                       background='#F8C471',  # Light orange background
                       foreground=self.colors['galileo_dark'],
                       font=('Arial', 10, 'bold'),
                       borderwidth=2,
                       relief='raised')
        
        style.map('Warning.TButton',
                  background=[('active', '#F5B041'), ('pressed', '#F4D03F')],  # Light orange/yellow shades
                  foreground=[('active', self.colors['galileo_dark']),
                             ('pressed', self.colors['galileo_dark'])])
        
        # Original professional button style
        style.configure('TButton',
                       font=('Arial', 10),
                       padding=(10, 5),
                       background=self.colors['galileo_white'],
                       foreground=self.colors['galileo_dark'],  # Dark text for default buttons
                       borderwidth=1,
                       relief='raised')
        
        style.map('TButton',
                  background=[('active', self.colors['galileo_light']),
                             ('pressed', '#D5DBDB')],
                  foreground=[('active', self.colors['galileo_dark']),
                             ('pressed', self.colors['galileo_dark'])])
    
    def create_config_screen(self):
        """Create initial configuration screen."""
        # Main config frame
        self.config_frame = ttk.Frame(self.root, padding="30")
        self.config_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(self.config_frame, text="🚀 Huey GPU Configuration", 
                               style='BrandedTitle.TLabel')
        title_label.pack(pady=(0, 30))
        
        # Configuration sections
        self.create_network_config()
        self.create_processing_config() 
        self.create_appearance_config()
        
        # Buttons
        button_frame = ttk.Frame(self.config_frame)
        button_frame.pack(pady=(30, 0))
        
        apply_btn = ttk.Button(button_frame, text="🚀 Start Huey GPU", 
                              command=self.apply_config, 
                              style='Branded.TButton')
        apply_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        cancel_btn = ttk.Button(button_frame, text="❌ Cancel", 
                               command=self.root.destroy)
        cancel_btn.pack(side=tk.LEFT)
    
    def create_network_config(self):
        """Create network configuration section."""
        network_frame = ttk.LabelFrame(self.config_frame, text="Network Settings", padding="15")
        network_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Max neurons
        ttk.Label(network_frame, text="Maximum Neurons:").grid(row=0, column=0, sticky=tk.W, pady=5)
        neurons_spinbox = ttk.Spinbox(network_frame, from_=50, to=500, width=10,
                                     textvariable=self.max_neurons)
        neurons_spinbox.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Conversation mode
        conv_check = ttk.Checkbutton(network_frame, text="🗨️ Conversation Mode (detect speakers)",
                                    variable=self.conversation_mode)
        conv_check.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)
    
    def create_processing_config(self):
        """Create processing configuration section."""
        proc_frame = ttk.LabelFrame(self.config_frame, text="Processing Settings", padding="15")
        proc_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Timeout
        ttk.Label(proc_frame, text="Timeout (hours):").grid(row=0, column=0, sticky=tk.W, pady=5)
        timeout_spinbox = ttk.Spinbox(proc_frame, from_=0.5, to=24.0, increment=0.5, width=10,
                                     textvariable=self.timeout_hours, format="%.1f")
        timeout_spinbox.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Exchange limit
        ttk.Label(proc_frame, text="Exchange Limit:").grid(row=1, column=0, sticky=tk.W, pady=5)
        exchange_spinbox = ttk.Spinbox(proc_frame, from_=100, to=50000, increment=100, width=10,
                                      textvariable=self.exchange_limit)
        exchange_spinbox.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
    
    def create_appearance_config(self):
        """Create appearance configuration section."""
        appearance_frame = ttk.LabelFrame(self.config_frame, text="Appearance Settings", padding="15")
        appearance_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Dark mode toggle
        dark_check = ttk.Checkbutton(appearance_frame, text="🌙 Dark Mode",
                                    variable=self.dark_mode,
                                    command=self.toggle_dark_mode)
        dark_check.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Info label
        info_label = ttk.Label(appearance_frame, 
                              text="Dark mode provides better visualization contrast for long analysis sessions.",
                              font=('Arial', 9, 'italic'))
        info_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
    
    def toggle_dark_mode(self):
        """Toggle between light and dark mode."""
        if self.dark_mode.get():
            self.apply_dark_theme()
        else:
            self.apply_light_theme()
        
        # Force a complete redraw of the interface
        self.root.update_idletasks()
    
    def apply_dark_theme(self):
        """Apply comprehensive dark theme colors."""
        # Update branding colors for dark mode
        self.branding.brand_colors.update({
            'galileo_light': '#2C3E50',  # Dark background
            'galileo_dark': '#ECF0F1',   # Light text
            'galileo_white': '#34495E',  # Dark white
            'galileo_blue': '#3498DB',   # Bright blue for dark mode
        })
        
        # Dark mode color scheme
        self.dark_colors = {
            'bg': '#2C3E50',        # Dark background
            'fg': '#ECF0F1',        # Light text
            'select_bg': '#3498DB', # Blue selection
            'select_fg': '#FFFFFF', # White selected text
            'insert_bg': '#ECF0F1', # Light cursor
            'entry_bg': '#34495E',  # Dark entry fields
            'button_bg': '#34495E', # Dark buttons
            'frame_bg': '#2C3E50',  # Dark frames
        }
        
        # Update root window background
        self.root.configure(bg=self.dark_colors['bg'])
        
        # Apply dark theme to all existing widgets
        self.apply_theme_to_widgets(self.root, is_dark=True)
        
        # Refresh styles
        self.setup_styles()
    
    def apply_light_theme(self):
        """Apply original professional light theme colors."""
        # Restore original Galileo branding colors
        self.branding.brand_colors.update({
            'galileo_light': '#ECF0F1',  # Original light background
            'galileo_dark': '#2C3E50',   # Original dark text
            'galileo_white': '#FFFFFF',  # Pure white
            'galileo_blue': '#1F4E79',   # Original dark blue for light mode
        })
        
        # Original light mode color scheme
        self.light_colors = {
            'bg': '#ECF0F1',        # Original Galileo light background
            'fg': '#2C3E50',        # Original Galileo dark text
            'select_bg': '#3498DB', # Original blue selection
            'select_fg': '#FFFFFF', # White selected text
            'insert_bg': '#2C3E50', # Dark cursor
            'entry_bg': '#FFFFFF',  # White entry fields
            'button_bg': '#FFFFFF', # White buttons
            'frame_bg': '#ECF0F1',  # Original light frames
        }
        
        # Update root window background  
        self.root.configure(bg=self.light_colors['bg'])
        
        # Apply light theme to all existing widgets
        self.apply_theme_to_widgets(self.root, is_dark=False)
        
        # Refresh styles
        self.setup_styles()
    
    def apply_theme_to_widgets(self, parent, is_dark=True):
        """Recursively apply theme to all widgets."""
        colors = self.dark_colors if is_dark else self.light_colors
        
        # Apply theme to all text widgets
        for widget in getattr(self, 'text_widgets', []):
            try:
                widget.configure(
                    bg=colors['entry_bg'],
                    fg=colors['fg'],
                    selectbackground=colors['select_bg'],
                    selectforeground=colors['select_fg'],
                    insertbackground=colors['insert_bg']
                )
            except tk.TclError:
                pass  # Widget might not support all options
        
        # Recursively apply to all child widgets
        for child in parent.winfo_children():
            try:
                widget_class = child.winfo_class()
                
                # Handle different widget types
                if widget_class == 'Text':
                    child.configure(
                        bg=colors['entry_bg'],
                        fg=colors['fg'],
                        selectbackground=colors['select_bg'],
                        selectforeground=colors['select_fg'],
                        insertbackground=colors['insert_bg']
                    )
                elif widget_class == 'Listbox':
                    child.configure(
                        bg=colors['entry_bg'],
                        fg=colors['fg'],
                        selectbackground=colors['select_bg'],
                        selectforeground=colors['select_fg']
                    )
                elif widget_class == 'Entry':
                    child.configure(
                        bg=colors['entry_bg'],
                        fg=colors['fg'],
                        selectbackground=colors['select_bg'],
                        selectforeground=colors['select_fg'],
                        insertbackground=colors['insert_bg']
                    )
                elif widget_class == 'Frame':
                    child.configure(bg=colors['frame_bg'])
                elif widget_class == 'Label':
                    # Only apply to regular tk.Label widgets, not ttk
                    if hasattr(child, 'configure') and 'bg' in child.keys():
                        child.configure(bg=colors['bg'], fg=colors['fg'])
                
                # Handle matplotlib figures
                if hasattr(child, 'get_tk_widget'):
                    # This is likely a matplotlib canvas
                    fig = child.figure
                    if is_dark:
                        fig.patch.set_facecolor('#2C3E50')
                        for ax in fig.axes:
                            ax.set_facecolor('#34495E')
                            ax.tick_params(colors='#ECF0F1')
                            ax.xaxis.label.set_color('#ECF0F1')
                            ax.yaxis.label.set_color('#ECF0F1')
                            ax.title.set_color('#ECF0F1')
                            # Update spine colors
                            for spine in ax.spines.values():
                                spine.set_color('#ECF0F1')
                    else:
                        fig.patch.set_facecolor('#ECF0F1')  # Original light background
                        for ax in fig.axes:
                            ax.set_facecolor('#FFFFFF')     # White plot area
                            ax.tick_params(colors='#2C3E50') # Original dark ticks
                            ax.xaxis.label.set_color('#2C3E50')
                            ax.yaxis.label.set_color('#2C3E50')
                            ax.title.set_color('#2C3E50')
                            # Original spine colors
                            for spine in ax.spines.values():
                                spine.set_color('#2C3E50')
                    
                    child.draw()
                
            except (tk.TclError, AttributeError):
                pass  # Skip widgets that don't support the configuration
            
            # Recurse into child widgets
            self.apply_theme_to_widgets(child, is_dark)
    
    def apply_config(self):
        """Apply configuration and switch to main interface."""
        self.config_complete = True
        
        # Hide config screen
        self.config_frame.destroy()
        
        # Create main interface
        self.create_widgets()
        
        # Initialize Huey with settings
        self.initialize_huey()
        
        print(f"✅ Configuration applied - Max neurons: {self.max_neurons.get()}, "
              f"Dark mode: {self.dark_mode.get()}, Conversation mode: {self.conversation_mode.get()}")

    def create_widgets(self):
        """Create and layout all GUI widgets."""
        
        # Main container with minimal padding for more space
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights - give more space to main content
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)  # More space for main content
        main_frame.rowconfigure(2, weight=1)
        
        # Header section
        self.create_header(main_frame)
        
        # Main content area (tabbed interface) - now takes full space without settings panel
        self.create_main_content(main_frame)
        
        # Status bar
        self.create_status_bar(main_frame)
    
    def create_header(self, parent):
        """Create the application header with branding."""
        
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        header_frame.columnconfigure(1, weight=1)
        
        # Load Galileo logo
        logo_frame = ttk.Frame(header_frame, relief='solid', borderwidth=1)
        logo_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.N, tk.S), padx=(0, 20))
        
        # Try to load custom logo, fall back to branded placeholder
        logo_img = self.branding.load_image('galileo_logo.png', (120, 80))
        if logo_img:
            logo_label = tk.Label(logo_frame, image=logo_img, 
                                 background=self.colors['galileo_white'])
            logo_label.image = logo_img  # Keep reference
            logo_label.pack(padx=10, pady=10)
        else:
            # Fallback text logo
            logo_label = ttk.Label(logo_frame, text="🚀\nGALILEO\nHUEY", 
                                  font=('Arial', 10, 'bold'),
                                  justify=tk.CENTER)
            logo_label.pack(padx=15, pady=10)
        
        # Title and subtitle
        title_label = ttk.Label(header_frame, text="Huey GPU", style='Header.TLabel')
        title_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        subtitle_label = ttk.Label(header_frame, 
                                  text="Hebbian Self-Concept Analysis Platform with GPU Acceleration",
                                  font=('Arial', 10))
        subtitle_label.grid(row=1, column=1, sticky=(tk.W, tk.E))
    
    
    def create_main_content(self, parent):
        """Create the main tabbed content area."""
        
        # Create notebook for tabs with branded styling - MUCH LARGER
        self.notebook = ttk.Notebook(parent, style='Branded.TNotebook')
        self.notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # File Processing tab
        self.create_file_processing_tab()
        
        # Results Analysis tab  
        self.create_results_tab()
        
        # Visualization tab
        self.create_visualization_tab()
        
        # Settings tab
        self.create_advanced_settings_tab()
    
    def create_file_processing_tab(self):
        """Create the file processing tab."""
        
        file_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(file_frame, text="📁 File Processing")
        
        file_frame.columnconfigure(0, weight=1)
        file_frame.rowconfigure(2, weight=1)
        
        # File selection
        file_select_frame = ttk.LabelFrame(file_frame, text="File Selection", padding="10")
        file_select_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_select_frame.columnconfigure(0, weight=1)
        
        self.file_path = tk.StringVar(value="No file selected")
        file_label = ttk.Label(file_select_frame, textvariable=self.file_path,
                              relief='sunken', padding="5")
        file_label.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        browse_btn = ttk.Button(file_select_frame, text="Browse...", 
                               command=self.browse_file, style='Primary.TButton')
        browse_btn.grid(row=0, column=1)
        
        # Processing controls
        control_frame = ttk.LabelFrame(file_frame, text="Processing Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.process_btn = ttk.Button(control_frame, text="🚀 Process File", 
                                     command=self.process_file, style='Success.TButton')
        self.process_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(control_frame, text="⏹️ Stop", 
                                  command=self.stop_processing, style='Warning.TButton',
                                  state='disabled')
        self.stop_btn.pack(side=tk.LEFT)
        
        # Progress and output
        output_frame = ttk.LabelFrame(file_frame, text="Processing Output", padding="10")
        output_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(1, weight=1)
        
        # Enhanced progress bar with time estimation
        progress_frame = ttk.Frame(output_frame)
        progress_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                          maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Progress percentage label
        self.progress_label = tk.StringVar(value="Ready")
        progress_text = ttk.Label(progress_frame, textvariable=self.progress_label,
                                 font=('Arial', 9, 'bold'),
                                 foreground=self.colors['galileo_blue'])
        progress_text.grid(row=0, column=1, sticky=tk.E, padx=(10, 0))
        
        # Output text area - SMALLER to give more space to results
        self.output_text = scrolledtext.ScrolledText(output_frame, height=10, width=60,
                                                    wrap=tk.WORD, font=('Courier', 9))
        self.output_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Store reference for theme updates
        self.text_widgets = getattr(self, 'text_widgets', [])
        self.text_widgets.append(self.output_text)
    
    def create_results_tab(self):
        """Create the results analysis tab."""
        
        results_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(results_frame, text="📊 Results")
        
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Summary metrics
        metrics_frame = ttk.LabelFrame(results_frame, text="Analysis Summary", padding="10")
        metrics_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Metrics grid
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill=tk.X)
        
        # Create metric labels
        self.metric_labels = {}
        metrics = ['Speakers', 'Exchanges', 'Concepts', 'Connections', 'Total Mass', 'Processing Time']
        
        for i, metric in enumerate(metrics):
            col = i % 3
            row = i // 3
            
            frame = ttk.Frame(metrics_grid)
            frame.grid(row=row, column=col, padx=10, pady=5, sticky=tk.W)
            
            ttk.Label(frame, text=f"{metric}:", font=('Arial', 9, 'bold')).pack(anchor=tk.W)
            self.metric_labels[metric] = ttk.Label(frame, text="--", 
                                                  font=('Arial', 12),
                                                  foreground=self.colors['galileo_blue'])
            self.metric_labels[metric].pack(anchor=tk.W)
        
        # Detailed results
        details_frame = ttk.LabelFrame(results_frame, text="Detailed Analysis", padding="10")
        details_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)
        
        # MUCH LARGER results display
        self.results_text = scrolledtext.ScrolledText(details_frame, height=25, width=100,
                                                     wrap=tk.WORD, font=('Courier', 10))
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Store reference for theme updates
        self.text_widgets = getattr(self, 'text_widgets', [])
        self.text_widgets.append(self.results_text)
    
    def create_visualization_tab(self):
        """Create the visualization tab with matplotlib integration."""
        
        viz_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(viz_frame, text="📈 Visualization")
        
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(1, weight=1)
        
        # Visualization controls
        viz_controls = ttk.LabelFrame(viz_frame, text="Visualization Options", padding="10")
        viz_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        viz_type = ttk.Combobox(viz_controls, values=['3D Network', 'Concept Clusters', 'Speaker Analysis'],
                               state='readonly', width=20)
        viz_type.set('3D Network')
        viz_type.pack(side=tk.LEFT, padx=(0, 10))
        
        refresh_btn = ttk.Button(viz_controls, text="🔄 Refresh", 
                               command=self.force_refresh_visualization)
        refresh_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        export_btn = ttk.Button(viz_controls, text="💾 Export", 
                              command=self.export_visualization)
        export_btn.pack(side=tk.LEFT)
        
        # Add status indicator for visualization
        self.viz_status = tk.StringVar(value="No data processed yet")
        status_label = ttk.Label(viz_controls, textvariable=self.viz_status, 
                                font=('Arial', 9, 'italic'),
                                foreground=self.colors['galileo_teal'])
        status_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Add debug info button
        debug_btn = ttk.Button(viz_controls, text="🔍 Debug", 
                              command=self.show_debug_info)
        debug_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Add interactive window button
        interactive_btn = ttk.Button(viz_controls, text="🗗 Interactive Window", 
                                   command=self.open_interactive_visualization)
        interactive_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # MUCH LARGER Matplotlib canvas for prominent visualization
        self.fig = Figure(figsize=(14, 10), dpi=100, facecolor=self.colors['galileo_light'])
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Connect zoom and pan events to main canvas
        self.setup_main_canvas_events()
    
    def setup_main_canvas_events(self):
        """Setup zoom and pan events for the main visualization canvas."""
        self.main_scroll_event = self.canvas.mpl_connect('scroll_event', self.main_on_scroll)
        self.main_press_event = self.canvas.mpl_connect('button_press_event', self.main_on_press)
        self.main_release_event = self.canvas.mpl_connect('button_release_event', self.main_on_release)
        self.main_drag_event = self.canvas.mpl_connect('motion_notify_event', self.main_on_drag)
    
    def create_advanced_settings_tab(self):
        """Create the advanced settings tab."""
        
        adv_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(adv_frame, text="⚙️ Advanced")
        
        # Appearance Settings  
        appearance_frame = ttk.LabelFrame(adv_frame, text="Appearance", padding="10")
        appearance_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Dark mode toggle
        ttk.Checkbutton(appearance_frame, text="🌙 Dark Mode",
                       variable=self.dark_mode,
                       command=self.toggle_dark_mode).pack(anchor=tk.W)
        
        # GPU Settings (simplified)
        gpu_frame = ttk.LabelFrame(adv_frame, text="GPU Status", padding="10")
        gpu_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Initialize gpu_enabled if not already set
        if not hasattr(self, 'gpu_enabled'):
            self.gpu_enabled = tk.BooleanVar(value=True)
            
        ttk.Checkbutton(gpu_frame, text="Enable GPU Acceleration", 
                       variable=self.gpu_enabled,
                       command=self.toggle_gpu).pack(anchor=tk.W)
        
        # Visualization Controls
        viz_frame = ttk.LabelFrame(adv_frame, text="Visualization", padding="10")
        viz_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(viz_frame, text="🎯 Open Interactive 3D Window", 
                  command=self.open_interactive_visualization).pack(side=tk.LEFT, padx=(0, 10))
        
        # Export Settings
        export_frame = ttk.LabelFrame(adv_frame, text="Export Options", padding="10")
        export_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(export_frame, text="📄 Export Results to JSON", 
                  command=self.export_json).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(export_frame, text="📊 Export to CSV", 
                  command=self.export_csv).pack(side=tk.LEFT, padx=(0, 10))
        
        # About section
        about_frame = ttk.LabelFrame(adv_frame, text="About", padding="10")
        about_frame.pack(fill=tk.X)
        
        about_text = """Huey GPU - Hebbian Self-Concept Analysis Platform
        
GPU-accelerated neural network analysis for conversational data.
Built with revolutionary JAX acceleration targeting O(n²) bottlenecks.

Copyright (c) 2025 Emary Iacobucci and Joseph Woelfel
Licensed under Galileo Research Framework"""
        
        ttk.Label(about_frame, text=about_text, font=('Arial', 9),
                 justify=tk.LEFT).pack(anchor=tk.W)
    
    def create_status_bar(self, parent):
        """Create the status bar at the bottom."""
        
        status_frame = ttk.Frame(parent, relief='sunken', borderwidth=1)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        status_frame.columnconfigure(0, weight=1)
        
        self.status_text = tk.StringVar(value="Ready - Initialize Huey network and load a file to begin")
        status_label = ttk.Label(status_frame, textvariable=self.status_text, 
                                font=('Arial', 9),
                                foreground='#2C3E50')  # Dark text for better contrast
        status_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        # GPU status with branded indicator
        gpu_frame = ttk.Frame(status_frame)
        gpu_frame.grid(row=0, column=1, sticky=tk.E, padx=5, pady=2)
        
        self.gpu_indicator = self.branding.create_status_indicator(gpu_frame, 'success', (16, 16))
        self.gpu_indicator.pack(side=tk.LEFT, padx=(0, 5))
        
        self.gpu_status = tk.StringVar(value="🚀 GPU")
        gpu_label = ttk.Label(gpu_frame, textvariable=self.gpu_status,
                             font=('Arial', 9, 'bold'),
                             foreground=self.colors['galileo_green'])
        gpu_label.pack(side=tk.LEFT)
    
    def initialize_huey(self):
        """Initialize the Huey GPU network."""
        try:
            # Initialize gpu_enabled if not already set
            if not hasattr(self, 'gpu_enabled'):
                self.gpu_enabled = tk.BooleanVar(value=True)
                
            self.huey = HueyGPUConversationalNetwork(
                max_neurons=self.max_neurons.get(),
                use_gpu_acceleration=self.gpu_enabled.get(),
                conversation_mode=self.conversation_mode.get()
            )
            if hasattr(self, 'status_text'):
                self.status_text.set("✅ Huey GPU network initialized successfully")
            self.log_message("🚀 Huey GPU network ready for analysis")
            
        except Exception as e:
            if hasattr(self, 'status_text'):
                self.status_text.set(f"❌ Failed to initialize Huey: {str(e)}")
            print(f"❌ Failed to initialize Huey: {str(e)}")  # Fallback
            messagebox.showerror("Initialization Error", f"Could not initialize Huey GPU:\n{str(e)}")
    
    def update_huey_settings(self):
        """Update Huey network settings."""
        if self.huey:
            self.huey.max_neurons = self.max_neurons.get()
            self.status_text.set("⚙️ Network settings updated")
    
    def browse_file(self):
        """Open file browser dialog."""
        file_types = [
            ('Text files', '*.txt'),
            ('PDF files', '*.pdf'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select conversation file",
            filetypes=file_types
        )
        
        if filename:
            self.file_path.set(filename)
            # Defensive check - make sure status_text exists
            if hasattr(self, 'status_text'):
                self.status_text.set(f"📁 File selected: {os.path.basename(filename)}")
            else:
                print(f"📁 File selected: {os.path.basename(filename)}")  # Fallback to console
    
    def process_file(self):
        """Process the selected file in a separate thread."""
        if not self.file_path.get() or self.file_path.get() == "No file selected":
            messagebox.showwarning("No File", "Please select a file to process.")
            return
        
        if not self.huey:
            messagebox.showerror("No Network", "Huey network not initialized.")
            return
        
        # Disable processing button and enable stop button
        self.process_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        # Clear previous output
        self.output_text.delete(1.0, tk.END)
        self.progress_var.set(0)
        
        # Start processing in separate thread
        self.processing_thread = threading.Thread(target=self._process_file_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start monitoring thread results
        self.root.after(100, self.check_processing_queue)
    
    def _process_file_thread(self):
        """Optimized file processing thread worker with minimal progress updates for performance."""
        try:
            filename = self.file_path.get()
            conversation_mode = self.conversation_mode.get()
            
            self.results_queue.put(('status', '🚀 Starting file processing...'))
            self.results_queue.put(('progress', 5))
            
            # Removed detailed progress callback to improve performance
                
            # Process file with enhanced progress tracking
            self.results_queue.put(('status', '📝 Analyzing file structure...'))
            self.results_queue.put(('progress', 10))
            
            from huey_speaker_detector import HueySpeakerDetector
            detector = HueySpeakerDetector(conversation_mode=conversation_mode)
            
            self.results_queue.put(('status', '🎤 Detecting speakers and exchanges...'))
            self.results_queue.put(('progress', 20))
            
            # Process conversation file
            result = detector.process_conversation_file(filename)
            
            if 'error' in result:
                self.results_queue.put(('error', result['error']))
                return
                
            self.results_queue.put(('status', f"📊 Found {len(result['speakers_info'])} speakers, {len(result['conversation_data'])} exchanges"))
            self.results_queue.put(('progress', 35))
            
            # Register speakers
            self.results_queue.put(('status', '👥 Registering speakers in neural network...'))
            self.results_queue.put(('progress', 40))
            
            for speaker_info in result['speakers_info']:
                speaker_id = speaker_info[0]
                self.huey.add_speaker(speaker_id, ['i', 'me', 'my', 'myself'], ['you', 'your', 'yours'])
                
            # Process exchanges with MINIMAL progress updates for performance
            total_exchanges = len(result['conversation_data'])
            self.results_queue.put(('status', f"🧠 Processing {total_exchanges} conversation exchanges..."))
            
            # Define progress checkpoints (only update at 10%, 25%, 50%, 75%, 90%)
            checkpoints = [int(total_exchanges * p) for p in [0.1, 0.25, 0.5, 0.75, 0.9]]
            
            for i, (speaker, text) in enumerate(result['conversation_data']):
                # ONLY update progress at major checkpoints to avoid slowdown
                if i in checkpoints:
                    progress = 40 + int((i / total_exchanges) * 50)
                    concepts = len(getattr(self.huey, 'concept_neurons', {}))
                    connections = len(getattr(self.huey, 'connections', {}))
                    self.results_queue.put(('progress', progress))
                    self.results_queue.put(('status', f"🔥 Exchange {i+1}/{total_exchanges} - {concepts} concepts, {connections} connections"))
                
                self.huey.process_speaker_text(speaker, text)
            
            # Final processing and concept mapping
            self.results_queue.put(('status', '🎨 Finalizing analysis and generating results...'))
            self.results_queue.put(('progress', 95))
            
            # Force final sync to capture all concepts for visualization
            if hasattr(self.huey, '_sync_network_mappings'):
                self.huey._sync_network_mappings()
            
            # Build final result
            final_result = {
                'success': True,
                'speakers_registered': len(result['speakers_info']),
                'exchanges_processed': len(result['conversation_data']),
                'detection_info': result['detection_info'],
                'conversation_mode': conversation_mode,
                'concepts_created': len(getattr(self.huey, 'concept_neurons', {})),
                'connections_formed': len(getattr(self.huey, 'connections', {})),
                'activations_tracked': len(getattr(self.huey, 'activations', {}))
            }
            
            self.results_queue.put(('progress', 100))
            self.results_queue.put(('status', '✅ File processed successfully!'))
            self.results_queue.put(('result', final_result))
                
        except Exception as e:
            self.results_queue.put(('error', str(e)))
    
    def check_processing_queue(self):
        """Check for messages from the processing thread."""
        try:
            while True:
                msg_type, data = self.results_queue.get_nowait()
                
                if msg_type == 'status':
                    self.status_text.set(data)
                    self.log_message(data)
                    
                elif msg_type == 'progress':
                    self.progress_var.set(data)
                    self.progress_label.set(f"{int(data)}%")
                    # Change color based on progress
                    if data >= 100:
                        self.progress_label.set("✅ Complete!")
                    elif data >= 90:
                        self.progress_label.set(f"🎯 {int(data)}%")
                    elif data >= 50:
                        self.progress_label.set(f"🔥 {int(data)}%")
                    else:
                        self.progress_label.set(f"⚡ {int(data)}%")
                    
                elif msg_type == 'result':
                    self.display_results(data)
                    
                    # UPDATE GLOBAL DATA STATE
                    self.global_data_state.update({
                        'data_processed': True,
                        'processing_complete': True,
                        'results_available': True,
                        'visualization_ready': True,
                        'last_update': time.time()
                    })
                    
                    self.processing_complete()
                    # Switch to results tab to show the results prominently
                    self.notebook.select(1)
                    
                    # Multiple visualization refresh attempts with increasing delays
                    self.root.after(100, self.refresh_visualization)   # First attempt
                    self.root.after(1000, self.refresh_visualization)  # Second attempt
                    self.root.after(2000, self.refresh_visualization)  # Final attempt
                    
                elif msg_type == 'error':
                    self.status_text.set(f"❌ Error: {data}")
                    self.log_message(f"❌ Error: {data}")
                    messagebox.showerror("Processing Error", data)
                    self.processing_complete()
                    
        except queue.Empty:
            pass
        
        # Continue monitoring if processing is still running
        if self.processing_thread and self.processing_thread.is_alive():
            self.root.after(100, self.check_processing_queue)
    
    def processing_complete(self):
        """Clean up after processing completes."""
        self.process_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.progress_var.set(100)
    
    def stop_processing(self):
        """Stop the processing thread."""
        if self.processing_thread:
            # Note: This is a simple implementation
            # For production, you'd need proper thread communication
            self.status_text.set("⏹️ Processing stopped by user")
            self.log_message("⏹️ Processing stopped by user")
            self.processing_complete()
    
    def display_results(self, results):
        """Display processing results in the GUI."""
        
        # Update metrics safely
        if 'Speakers' in self.metric_labels:
            self.metric_labels['Speakers'].config(text=str(results.get('speakers_registered', '--')))
        if 'Exchanges' in self.metric_labels:
            self.metric_labels['Exchanges'].config(text=str(results.get('exchanges_processed', '--')))
        
        if self.huey:
            if 'Concepts' in self.metric_labels:
                concept_count = len(getattr(self.huey, 'concept_neurons', {}))
                self.metric_labels['Concepts'].config(text=str(concept_count))
            if 'Connections' in self.metric_labels:
                connection_count = len(getattr(self.huey, 'connections', {}))
                self.metric_labels['Connections'].config(text=str(connection_count))
            if 'Total Mass' in self.metric_labels:
                # Calculate total mass from all concept activations and connections
                total_mass = 0.0
                for concept, neuron_id in self.huey.concept_neurons.items():
                    activation = self.huey.activations.get(neuron_id, 0.0)
                    conn_count = sum(1 for key in self.huey.connections.keys() if neuron_id in key)
                    concept_mass = activation * 0.5 + conn_count * 0.1
                    total_mass += concept_mass
                
                # Update the network's inertial_mass dictionary
                if hasattr(self.huey, 'inertial_mass'):
                    self.huey.inertial_mass.clear()
                    for concept, neuron_id in self.huey.concept_neurons.items():
                        activation = self.huey.activations.get(neuron_id, 0.0)
                        conn_count = sum(1 for key in self.huey.connections.keys() if neuron_id in key)
                        self.huey.inertial_mass[neuron_id] = activation * 0.5 + conn_count * 0.1
                
                self.metric_labels['Total Mass'].config(text=f"{total_mass:.2f}")
            if 'Processing Time' in self.metric_labels:
                # Get GPU performance stats
                gpu_stats = self.huey.gpu_interface.get_performance_stats()
                proc_time = gpu_stats.get('total_kernel_time', 0.0)
                self.metric_labels['Processing Time'].config(text=f"{proc_time:.3f}s")
        
        # Display detailed results
        self.results_text.delete(1.0, tk.END)
        
        results_text = f"""HUEY GPU ANALYSIS RESULTS
{'=' * 50}

File Processing:
• Speakers Registered: {results.get('speakers_registered', 'N/A')}
• Exchanges Processed: {results.get('exchanges_processed', 'N/A')}
• Conversation Mode: {results.get('conversation_mode', 'N/A')}

Network State:"""
        
        if self.huey:
            results_text += f"""
• Total Neurons: {self.huey.neuron_count}
• Active Connections: {len(self.huey.connections)}
• Concept Vocabulary: {len(self.huey.concept_neurons)}

GPU Performance:
{self.huey.get_performance_summary()}
"""
        
        self.results_text.insert(tk.END, results_text)
        
        # Switch to results tab
        self.notebook.select(1)
    
    def refresh_visualization(self):
        """Refresh the visualization display."""
        if not self.huey:
            self.show_no_data_message("Initialize Huey network first")
            return
        
        # COMPREHENSIVE data availability check using global state
        data_available = (
            self.global_data_state['data_processed'] and
            self.global_data_state['results_available'] and
            hasattr(self.huey, 'concept_neurons') and 
            len(self.huey.concept_neurons) > 0
        )
        
        # Update visualization status
        if hasattr(self, 'viz_status'):
            if data_available:
                concept_count = len(self.huey.concept_neurons)
                connection_count = len(getattr(self.huey, 'connections', {}))
                self.viz_status.set(f"✅ Ready: {concept_count} concepts, {connection_count} connections")
            else:
                self.viz_status.set("❌ No data - process a file first")
        
        if not data_available:
            self.show_no_data_message("Process a file first to generate visualizations")
            return
        
        # Apply professional Seaborn styling
        sns.set_style("darkgrid" if hasattr(self, 'dark_mode') and self.dark_mode.get() else "whitegrid")
        sns.set_palette("plasma")  # Use plasma palette to match Streamlit
        
        # Set matplotlib parameters for publication quality
        plt.rcParams.update({
            'font.size': 11,
            'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'figure.facecolor': self.colors.get('galileo_light', 'white'),
            'axes.facecolor': self.colors.get('galileo_light', 'white')
        })
        
        self.fig.clear()
        
        try:
            # Use sophisticated Galileo visualization (same as Streamlit) with debug
            print(f"🔍 DEBUG: Starting sophisticated 3D plot generation...")
            result = self.create_sophisticated_3d_plot(num_concepts=50, min_mass=0.0001)  # Lower threshold
            
            if result is not None:
                positions, concept_data, warp_factor = result
                
                if len(positions) > 0:
                    ax = self.fig.add_subplot(111, projection='3d')
                    
                    # Extract coordinates
                    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
                    masses = [c['mass'] for c in concept_data]
                    labels = [c['name'] for c in concept_data]
                    
                    # Professional 3D styling with Seaborn-inspired colors
                    plasma_colors = sns.color_palette("plasma", as_cmap=True)
                    
                    # Create scatter plot with enhanced styling
                    masses_array = np.array(masses)
                    sizes = masses_array * 300 + 80  # Refined sizing
                    normalized_masses = (masses_array - masses_array.min()) / (masses_array.max() - masses_array.min()) if masses_array.max() > masses_array.min() else np.ones_like(masses_array)
                    
                    scatter = ax.scatter(x, y, z, 
                                       c=normalized_masses, 
                                       s=sizes,
                                       cmap=plasma_colors, 
                                       alpha=0.8, 
                                       edgecolors='white', 
                                       linewidth=1.2,
                                       depthshade=True)
                    
                    # Add subtle glow effect for important concepts
                    for i, (xi, yi, zi, mass, label) in enumerate(zip(x, y, z, masses, labels)):
                        if mass > np.percentile(masses, 75):  # Top 25% by mass
                            ax.scatter(xi, yi, zi, 
                                     c=plasma_colors(normalized_masses[i]), 
                                     s=sizes[i] * 1.5, 
                                     alpha=0.3, 
                                     edgecolors='none')
                    
                    # Enhanced labels with professional styling
                    top_indices = np.argsort(masses)[-12:][::-1]  # Top 12 by mass
                    for idx in top_indices:
                        label = labels[idx]
                        # Create text with shadow effect
                        text = ax.text(x[idx], y[idx], z[idx], label, 
                                     fontsize=9, fontweight='bold', 
                                     color='white',
                                     bbox=dict(boxstyle="round,pad=0.3", 
                                              facecolor=plasma_colors(normalized_masses[idx]), 
                                              alpha=0.9, 
                                              edgecolor='white',
                                              linewidth=1.5))
                        # Add text shadow effect
                        text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])
                    
                    # Professional axis styling
                    ax.set_xlabel('Semantic Dimension I', fontsize=12, fontweight='bold', 
                                 color=self.colors.get('galileo_dark', 'black'))
                    ax.set_ylabel('Semantic Dimension II', fontsize=12, fontweight='bold',
                                 color=self.colors.get('galileo_dark', 'black')) 
                    ax.set_zlabel('Semantic Dimension III', fontsize=12, fontweight='bold',
                                 color=self.colors.get('galileo_dark', 'black'))
                    
                    # Enhanced title with warp factor
                    if warp_factor == float('inf'):
                        warp_str = "∞ (perfectly balanced)"
                        warp_color = 'gold'
                    elif warp_factor < 0:
                        warp_str = f"{warp_factor:.3f} (negative space)"
                        warp_color = 'red'
                    else:
                        warp_str = f"{warp_factor:.3f}"
                        warp_color = 'green'
                    
                    title = ax.set_title(f'Huey GPU: Galileo Concept Space\nWarp Factor: {warp_str}', 
                                        fontsize=13, fontweight='bold', pad=25,
                                        color=self.colors.get('galileo_dark', 'black'))
                    
                    # Professional colorbar
                    colorbar = self.fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.05, aspect=30)
                    colorbar.set_label('Inertial Mass', fontsize=11, fontweight='bold',
                                     color=self.colors.get('galileo_dark', 'black'))
                    colorbar.ax.tick_params(labelsize=9)
                    
                    # Set professional grid and background
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.xaxis.pane.fill = False
                    ax.yaxis.pane.fill = False
                    ax.zaxis.pane.fill = False
                    
                    # Make pane edges more subtle
                    ax.xaxis.pane.set_edgecolor('gray')
                    ax.yaxis.pane.set_edgecolor('gray')  
                    ax.zaxis.pane.set_edgecolor('gray')
                    ax.xaxis.pane.set_alpha(0.1)
                    ax.yaxis.pane.set_alpha(0.1)
                    ax.zaxis.pane.set_alpha(0.1)
                    
                    print(f"✅ Sophisticated visualization: {len(concept_data)} concepts, warp factor: {warp_str}")
                    
                else:
                    self.show_no_data_message("No concepts with sufficient mass found")
            else:
                self.show_no_data_message("Error generating sophisticated visualization")
            
        except Exception as e:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Visualization error:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14,
                   fontweight='bold', color=self.colors['galileo_red'])
            ax.set_title('🚀 Huey GPU - Visualization Error', fontsize=16, fontweight='bold')
        
        self.canvas.draw()
    
    def create_sophisticated_3d_plot(self, num_concepts=50, min_mass=0.001):
        """
        Create sophisticated 3D visualization using pseudo-Riemannian embedding.
        
        This implements the same algorithm as the working Streamlit interface.
        """
        import numpy as np
        
        try:
            # Debug network state
            print(f"🔍 DEBUG: Sophisticated 3D plot called")
            print(f"🔍 DEBUG: Huey has {len(getattr(self.huey, 'neuron_to_word', {}))} neurons")
            print(f"🔍 DEBUG: Huey has {len(getattr(self.huey, 'inertial_mass', {}))} inertial masses")
            print(f"🔍 DEBUG: Min mass threshold: {min_mass}")
            
            # System artifacts to filter out
            system_artifacts = {
                'speaker_speaker_a', 'speaker_speaker_b', 'speaker_speaker_c', 
                'speaker_speaker_d', 'speaker_speaker_e', 'speaker_speaker_f',
                're', 'e', 'g', '4', 'lines'
            }
            
            # Get all concepts and their masses
            all_concepts = []
            
            # Check if this is plain text mode
            plain_text_mode = False
            if hasattr(self.huey, 'speakers') and len(self.huey.speakers) == 1:
                speaker_names = [s.lower() for s in self.huey.speakers.keys()]
                if 'text' in speaker_names:
                    plain_text_mode = True
            
            # Get speaker masses first (if available)
            speaker_masses_found = False
            if hasattr(self.huey, 'speakers') and self.huey.speakers and not plain_text_mode:
                for speaker in self.huey.speakers:
                    if hasattr(self.huey, 'analyze_speaker_self_concept'):
                        analysis = self.huey.analyze_speaker_self_concept(speaker)
                        mass = analysis.get('self_concept_mass', 0.0)
                        if mass > 0:
                            # Use actual speaker neuron ID
                            speaker_neuron_word = speaker.lower()
                            actual_speaker_id = self.huey.word_to_neuron.get(speaker_neuron_word)
                            if actual_speaker_id is not None:
                                all_concepts.append({
                                    'name': f"Speaker_{speaker}",
                                    'mass': mass,
                                    'id': actual_speaker_id
                                })
                                speaker_masses_found = True
            
            # Get regular concept masses from individual neurons
            for neuron_id, word in self.huey.neuron_to_word.items():
                # Skip system artifacts
                if not plain_text_mode and word.lower() in system_artifacts:
                    continue
                if plain_text_mode and word.lower() in {'speaker_text', 're', 'e', 'g', '4', 'lines'}:
                    continue
                    
                # Calculate concept mass from inertial_mass connections
                total_mass = 0.0
                if hasattr(self.huey, 'inertial_mass'):
                    for (i, j), mass in self.huey.inertial_mass.items():
                        if i == neuron_id or j == neuron_id:
                            total_mass += mass
                
                # Debug mass calculation
            if neuron_id < 10:  # Debug first 10 neurons
                print(f"🔍 DEBUG: Concept '{word}' (id={neuron_id}) has mass={total_mass:.6f}, min_mass={min_mass}")
            
            # Only include concepts with significant mass
            if total_mass >= min_mass:
                all_concepts.append({
                    'name': word,
                    'mass': total_mass,
                    'id': neuron_id
                })
            
            # Sort by mass and take top concepts
            all_concepts.sort(key=lambda x: x['mass'], reverse=True)
            concepts_data = all_concepts[:num_concepts]
            
            print(f"🔍 DEBUG: Found {len(all_concepts)} concepts total, using top {len(concepts_data)}")
            if len(concepts_data) > 0:
                print(f"🔍 DEBUG: Top concept: '{concepts_data[0]['name']}' with mass {concepts_data[0]['mass']:.6f}")
                if len(concepts_data) > 1:
                    print(f"🔍 DEBUG: 2nd concept: '{concepts_data[1]['name']}' with mass {concepts_data[1]['mass']:.6f}")
            
            if len(concepts_data) < 3:
                print(f"❌ Only found {len(concepts_data)} concepts total (need at least 3)")
                return None
            
            # Build association matrix using SYMMETRIC approach (same as working connection plot)
            n = len(concepts_data)
            concept_ids = [c['id'] for c in concepts_data]
            association_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        nid_i, nid_j = concept_ids[i], concept_ids[j]
                        # Use symmetric connections for robust positioning
                        conn_key1 = (nid_i, nid_j)
                        conn_key2 = (nid_j, nid_i)
                        
                        mass = 0.0
                        if hasattr(self.huey, 'inertial_mass'):
                            if conn_key1 in self.huey.inertial_mass:
                                mass += self.huey.inertial_mass[conn_key1]
                            if conn_key2 in self.huey.inertial_mass:
                                mass += self.huey.inertial_mass[conn_key2]
                        
                        association_matrix[i, j] = mass
            
            # Convert to distance matrix and apply Torgerson double-centering
            max_sim = np.max(association_matrix) if np.max(association_matrix) > 0 else 1.0
            distance_matrix = max_sim - association_matrix
            
            # Double centering for Torgerson transformation
            H = np.eye(n) - np.ones((n, n)) / n
            B = -0.5 * H @ (distance_matrix**2) @ H
            
            eigenvalues, eigenvectors = np.linalg.eigh(B)
            idx = np.argsort(eigenvalues)[::-1]
            
            # Take top 3 dimensions with proper pseudo-Riemannian scaling
            selected_eigenvals = eigenvalues[idx[:3]]
            selected_eigenvecs = eigenvectors[:, idx[:3]]
            
            # Proper coordinate calculation preserving pseudo-Riemannian structure
            positions = np.zeros((n, 3))
            for i in range(3):
                eigenval = selected_eigenvals[i]
                eigenvec = selected_eigenvecs[:, i]
                if eigenval > 1e-8:
                    # Positive eigenvalue: standard scaling
                    positions[:, i] = eigenvec * np.sqrt(eigenval)
                elif eigenval < -1e-8:
                    # Negative eigenvalue: preserve negative metric signature
                    positions[:, i] = eigenvec * np.sqrt(-eigenval)
                else:
                    # Near-zero eigenvalue: minimal scaling
                    positions[:, i] = eigenvec * 1e-4
            
            positions *= 3  # Scale up for better visual separation
            
            # Calculate warp factor: sum of positive eigenvalues / sum of all eigenvalues
            positive_eigenvals = eigenvalues[eigenvalues > 1e-10]
            negative_eigenvals = eigenvalues[eigenvalues < -1e-10]
            
            if len(positive_eigenvals) > 0:
                sum_positive = np.sum(positive_eigenvals)
                sum_negative = np.sum(negative_eigenvals)  # This will be negative
                sum_all = sum_positive + sum_negative
                
                if abs(sum_all) > 1e-8:
                    warp_factor = sum_positive / sum_all
                else:
                    # Perfectly balanced space
                    warp_factor = float('inf')
            else:
                warp_factor = 1.0
            
            return positions, concepts_data, warp_factor
            
        except Exception as e:
            print(f"Error in sophisticated 3D plot: {e}")
            return None
    
    def force_refresh_visualization(self):
        """Force refresh visualization regardless of state - for manual refresh button."""
        if not self.huey:
            self.show_no_data_message("Initialize Huey network first")
            return
        
        # Check if we have ANY concept data
        if hasattr(self.huey, 'concept_neurons') and len(self.huey.concept_neurons) > 0:
            # Force update global state
            self.global_data_state.update({
                'data_processed': True,
                'results_available': True,
                'visualization_ready': True,
                'last_update': time.time()
            })
            # Force visualization refresh
            self.refresh_visualization()
        else:
            self.show_no_data_message("No concept data available - process a file first")
    
    def show_no_data_message(self, message: str):
        """Show a message when no data is available for visualization."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Show debug info about global state
        concept_count = len(getattr(self.huey, 'concept_neurons', {}))
        connection_count = len(getattr(self.huey, 'connections', {}))
        activation_count = len(getattr(self.huey, 'activations', {}))
        
        debug_info = f"""\n\n--- Debug Info ---
Global State: {self.global_data_state}
Huey Concepts: {concept_count}
Huey Connections: {connection_count}
Huey Activations: {activation_count}

Sample Concepts: {list(getattr(self.huey, 'concept_neurons', {}).keys())[:10]}"""
        
        full_message = message + debug_info
        
        ax.text(0.5, 0.5, full_message, 
               ha='center', va='center', transform=ax.transAxes, 
               fontsize=12, fontweight='bold', color=self.colors['galileo_dark'],
               bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['galileo_light'], 
                        edgecolor=self.colors['galileo_blue'], linewidth=2))
        ax.set_title('🚀 Huey GPU - 3D Concept Space Visualization', 
                    color=self.colors['galileo_blue'], fontsize=18, fontweight='bold', pad=20)
        ax.set_facecolor(self.colors['galileo_white'])
        ax.axis('off')  # Hide axes for cleaner look
        self.canvas.draw()
    
    def export_visualization(self):
        """Export the current visualization."""
        filename = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG files', '*.png'), ('PDF files', '*.pdf')]
        )
        
        if filename:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            self.status_text.set(f"📊 Visualization exported to {os.path.basename(filename)}")
    
    def export_json(self):
        """Export results to JSON."""
        if not self.huey:
            messagebox.showinfo("No Data", "No analysis results to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension='.json',
            filetypes=[('JSON files', '*.json')]
        )
        
        if filename:
            try:
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'network_stats': {
                        'neurons': self.huey.neuron_count,
                        'connections': len(self.huey.connections),
                        'concepts': len(self.huey.concept_neurons)
                    },
                    'concepts': list(self.huey.concept_neurons.keys()),
                    'gpu_performance': self.huey.get_performance_stats()
                }
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                self.status_text.set(f"💾 Results exported to {os.path.basename(filename)}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export JSON: {str(e)}")
    
    def export_csv(self):
        """Export concept data to CSV."""
        if not self.huey or not self.huey.concept_neurons:
            messagebox.showinfo("No Data", "No concept data to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv')]
        )
        
        if filename:
            try:
                # Create DataFrame with concept data
                data = []
                for concept, neuron_id in self.huey.concept_neurons.items():
                    activation = self.huey.activations.get(neuron_id, 0.0)
                    data.append({
                        'concept': concept,
                        'neuron_id': neuron_id,
                        'activation': activation
                    })
                
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
                
                self.status_text.set(f"📊 Concept data exported to {os.path.basename(filename)}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export CSV: {str(e)}")
    
    def toggle_gpu(self):
        """Toggle GPU acceleration."""
        if self.gpu_enabled.get():
            self.gpu_status.set("🚀 GPU")
            self.status_text.set("GPU acceleration enabled")
        else:
            self.gpu_status.set("💻 CPU")
            self.status_text.set("Using CPU fallback mode")
        
        # Reinitialize Huey with new GPU setting
        self.initialize_huey()
    
    def monitor_global_state(self):
        """Monitor global data state and update tabs accordingly."""
        # Check if visualization needs refreshing
        if (self.global_data_state['visualization_ready'] and 
            self.global_data_state['data_processed'] and
            hasattr(self.huey, 'concept_neurons') and
            len(self.huey.concept_neurons) > 0):
            
            # Update status to show data is ready
            if hasattr(self, 'status_text'):
                concept_count = len(self.huey.concept_neurons)
                connection_count = len(getattr(self.huey, 'connections', {}))
                self.status_text.set(f"✅ Ready: {concept_count} concepts, {connection_count} connections - Visualization available")
        
        # Schedule next check
        self.root.after(2000, self.monitor_global_state)
    
    def show_debug_info(self):
        """Show debug information about the current state."""
        debug_msg = f"""HUEY GPU DEBUG INFO:
        
Global State:
{json.dumps(self.global_data_state, indent=2, default=str)}

Huey Network State:
- Network initialized: {self.huey is not None}
- Concept neurons: {len(getattr(self.huey, 'concept_neurons', {}))}
- Connections: {len(getattr(self.huey, 'connections', {}))}
- Activations: {len(getattr(self.huey, 'activations', {}))}

Concepts: {list(getattr(self.huey, 'concept_neurons', {}).keys())[:10]}...
        """
        
        messagebox.showinfo("Debug Information", debug_msg)
    
    def open_interactive_visualization(self):
        """Open a separate interactive visualization window."""
        if not self.huey:
            messagebox.showwarning("No Network", "Initialize Huey network first.")
            return
            
        # Check if we have concept data
        if not hasattr(self.huey, 'concept_neurons') or len(self.huey.concept_neurons) == 0:
            messagebox.showwarning("No Data", "Process a file first to generate visualizations.")
            return
        
        # Create interactive visualization window
        InteractiveVisualizationWindow(self.huey, self.colors)
    
    def log_message(self, message):
        """Log a message to the output area."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.output_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.output_text.see(tk.END)
    
    # Main visualization canvas zoom and pan methods
    def main_on_scroll(self, event):
        """Handle mouse scroll events for zooming in main canvas."""
        if not hasattr(event, 'inaxes') or event.inaxes is None:
            return
            
        # Determine zoom direction
        zoom_in = event.step > 0
        zoom_factor = 1.2 if zoom_in else 1/1.2
        
        # Get current axis limits  
        ax = event.inaxes
        if hasattr(ax, 'get_zlim'):  # 3D plot
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()
            
            # Calculate zoom center
            if hasattr(event, 'xdata') and event.xdata is not None:
                center_x, center_y = event.xdata, event.ydata
            else:
                center_x = (xlim[0] + xlim[1]) / 2
                center_y = (ylim[0] + ylim[1]) / 2
            
            center_z = (zlim[0] + zlim[1]) / 2
            
            # Calculate new limits
            x_range = (xlim[1] - xlim[0]) / zoom_factor
            y_range = (ylim[1] - ylim[0]) / zoom_factor
            z_range = (zlim[1] - zlim[0]) / zoom_factor
            
            new_xlim = [center_x - x_range/2, center_x + x_range/2]
            new_ylim = [center_y - y_range/2, center_y + y_range/2]
            new_zlim = [center_z - z_range/2, center_z + z_range/2]
            
            # Store original limits if not already stored
            if self.main_original_xlim is None:
                self.main_original_xlim = xlim
                self.main_original_ylim = ylim
                self.main_original_zlim = zlim
            
            # Apply new limits
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
            ax.set_zlim(new_zlim)
            
            # Update zoom level
            self.main_zoom_level *= zoom_factor
            
            # Update canvas
            self.canvas.draw()
    
    def main_on_press(self, event):
        """Handle mouse button press for panning in main canvas."""
        if not hasattr(event, 'inaxes') or event.inaxes is None:
            return
            
        if event.button == 2:  # Middle mouse button for panning
            self.main_pan_start = (event.xdata, event.ydata)
            self.main_is_panning = True
    
    def main_on_release(self, event):
        """Handle mouse button release in main canvas."""
        if event.button == 2:  # Middle mouse button
            self.main_is_panning = False
            self.main_pan_start = None
    
    def main_on_drag(self, event):
        """Handle mouse drag events for panning in main canvas."""
        if not self.main_is_panning or self.main_pan_start is None:
            return
        
        if hasattr(event, 'inaxes') and event.inaxes is not None and hasattr(event, 'xdata') and event.xdata is not None:
            ax = event.inaxes
            
            # Calculate pan delta
            dx = event.xdata - self.main_pan_start[0]
            dy = event.ydata - self.main_pan_start[1]
            
            # Get current limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Apply pan (move in opposite direction of drag)
            new_xlim = [xlim[0] - dx, xlim[1] - dx]
            new_ylim = [ylim[0] - dy, ylim[1] - dy]
            
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
            
            # Update pan start position for continuous dragging
            self.main_pan_start = (event.xdata, event.ydata)
            
            # Update canvas
            self.canvas.draw()
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()

class InteractiveVisualizationWindow:
    """Separate interactive visualization window with full 3D controls."""
    
    def __init__(self, huey_network, colors):
        """Initialize the interactive visualization window."""
        self.huey = huey_network
        self.colors = colors
        
        # Interactive features
        self.tooltip_text = None
        self.selected_point = None
        self.scatter_plot = None
        self.coords_data = None
        self.labels_data = None
        
        # Pan and zoom state
        self.zoom_level = 1.0
        self.pan_start = None
        self.is_panning = False
        self.original_xlim = None
        self.original_ylim = None
        self.original_zlim = None
        
        # Create new window
        self.window = tk.Toplevel()
        self.window.title("🚀 Huey GPU - Interactive 3D Visualization")
        self.window.geometry("1400x900")
        self.window.minsize(800, 600)
        
        # Configure window
        self.setup_window()
        
        # Create visualization
        self.create_interactive_plot()
        
        # Initial data load
        self.refresh_interactive_plot()
        
        print("🎯 Interactive visualization window opened!")
    
    def setup_window(self):
        """Setup the interactive window layout."""
        # Main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Interactive Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Concept limit control
        ttk.Label(control_frame, text="Max Concepts:").pack(side=tk.LEFT, padx=(0, 5))
        self.concept_limit = tk.IntVar(value=20)
        concept_spinbox = ttk.Spinbox(control_frame, from_=5, to=100, width=8,
                                     textvariable=self.concept_limit, 
                                     command=self.refresh_interactive_plot)
        concept_spinbox.pack(side=tk.LEFT, padx=(0, 20))
        
        # Refresh button
        refresh_btn = ttk.Button(control_frame, text="🔄 Refresh Plot", 
                               command=self.refresh_interactive_plot)
        refresh_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Reset view button
        reset_btn = ttk.Button(control_frame, text="📍 Reset View", 
                             command=self.reset_view)
        reset_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Save image button
        save_btn = ttk.Button(control_frame, text="💾 Save Image", 
                            command=self.save_image)
        save_btn.pack(side=tk.LEFT, padx=(0, 20))
        
        # Concept cluster selection
        ttk.Label(control_frame, text="Highlight Concept:").pack(side=tk.LEFT, padx=(0, 5))
        self.selected_concept = tk.StringVar(value="None")
        self.concept_dropdown = ttk.Combobox(control_frame, textvariable=self.selected_concept,
                                           width=15, state="readonly")
        self.concept_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        self.concept_dropdown.bind('<<ComboboxSelected>>', self.on_concept_selected)
        
        # Status label
        self.status_var = tk.StringVar(value="Initializing...")
        status_label = ttk.Label(control_frame, textvariable=self.status_var, 
                                font=('Arial', 9, 'italic'),
                                foreground=self.colors['galileo_teal'])
        status_label.pack(side=tk.RIGHT)
        
        # Concept cluster analysis panel
        cluster_frame = ttk.LabelFrame(main_frame, text="Concept Cluster Analysis", padding="10")
        cluster_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Concept details display
        self.concept_details = tk.Text(cluster_frame, height=4, wrap=tk.WORD, 
                                     font=('Consolas', 9), state=tk.DISABLED)
        self.concept_details.pack(fill=tk.X)
        
        # Store reference for theme updates
        self.text_widgets = getattr(self, 'text_widgets', [])
        self.text_widgets.append(self.concept_details)
        
        # Visualization frame
        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure with toolbar
        self.fig = plt.figure(figsize=(16, 10), facecolor='white')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add matplotlib toolbar for interaction
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar = NavigationToolbar2Tk(self.canvas, viz_frame)
        self.toolbar.update()
        
        # Connect mouse events for hover tooltips (must be done after canvas is created)
        self.motion_event = None
        self.click_event = None
    
    def create_interactive_plot(self):
        """Create the interactive 3D plot."""
        self.fig.clear()
        
        # Create 3D subplot
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set up the plot
        self.ax.set_title('🚀 Huey GPU - Interactive 3D Concept Space', 
                         fontsize=16, fontweight='bold', pad=20)
        self.ax.set_xlabel('Semantic Dimension 1', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Semantic Dimension 2', fontsize=12, fontweight='bold')
        self.ax.set_zlabel('Semantic Dimension 3', fontsize=12, fontweight='bold')
    
    def refresh_interactive_plot(self):
        """Refresh the interactive plot with current data."""
        if not self.huey or not hasattr(self.huey, 'concept_neurons'):
            self.status_var.set("❌ No Huey data available")
            return
            
        concept_count = len(self.huey.concept_neurons)
        if concept_count == 0:
            self.status_var.set("❌ No concepts processed yet")
            return
        
        try:
            self.status_var.set("🔄 Generating sophisticated 3D coordinates...")
            self.window.update()
            
            # Use sophisticated visualization method from main GUI
            # Create a temporary reference to the sophisticated method
            parent_gui = self.parent if hasattr(self, 'parent') else None
            if parent_gui and hasattr(parent_gui, 'create_sophisticated_3d_plot'):
                result = parent_gui.create_sophisticated_3d_plot(num_concepts=self.concept_limit.get(), min_mass=0.001)
            else:
                # Fallback: inline sophisticated method
                result = self.create_inline_sophisticated_3d_plot(num_concepts=self.concept_limit.get(), min_mass=0.001)
            
            if result is None:
                self.status_var.set("❌ No sophisticated coordinate data generated")
                return
                
            positions, concept_data, warp_factor = result
            
            if len(positions) == 0:
                self.status_var.set("❌ No concepts with sufficient mass found")
                return
            
            # Clear and recreate plot
            self.create_interactive_plot()
            
            # Use sophisticated data
            coords_subset = positions
            labels_subset = [c['name'] for c in concept_data]
            concept_masses = [c['mass'] for c in concept_data]
            
            # Store data for interactive features
            self.coords_data = coords_subset
            self.labels_data = labels_subset
            
            # Use sophisticated masses and calculate connection counts
            connection_counts = []
            for concept in concept_data:
                neuron_id = concept['id']
                conn_count = sum(1 for key in self.huey.connections.keys() if neuron_id in key)
                connection_counts.append(conn_count)
            
            # Create scatter plot with sophisticated mass-based sizing and coloring
            sizes = np.array(concept_masses) * 500 + 50  # Scale masses for visual sizing
            colors = concept_masses  # Use masses for coloring
            
            self.scatter_plot = self.ax.scatter(coords_subset[:, 0], coords_subset[:, 1], coords_subset[:, 2],
                                    c=colors, s=sizes, cmap='plasma', 
                                    alpha=0.7, edgecolors='black', linewidth=1.5)
            
            # Add concept labels with intelligent positioning to avoid overlaps
            label_positions = self.calculate_label_positions(coords_subset, labels_subset)
            
            for i, (label, pos) in enumerate(zip(labels_subset, label_positions)):
                # Only show labels for top concepts to avoid clutter
                if i < 15 or concept_masses[i] > np.mean(concept_masses):
                    self.ax.text(pos[0], pos[1], pos[2], 
                               label, fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", 
                                       facecolor='white', alpha=0.9, edgecolor='gray'),
                               ha='center', va='bottom')
            
            # Store masses and connections for tooltips
            self.concept_masses = concept_masses
            self.connection_counts = connection_counts
            
            # Add connection visualization between related concepts
            self.draw_concept_connections(coords_subset, labels_subset)
            
            # Add colorbar
            colorbar = self.fig.colorbar(self.scatter_plot, ax=self.ax, shrink=0.6, pad=0.1)
            colorbar.set_label('Concept Activation Strength', fontsize=12, fontweight='bold')
            
            # Update concept dropdown with available concepts
            self.update_concept_dropdown(labels_subset)
            
            # Update status with warp factor
            connection_count = len(getattr(self.huey, 'connections', {}))
            if warp_factor == float('inf'):
                warp_str = "∞ (perfectly balanced)"
            elif warp_factor < 0:
                warp_str = f"{warp_factor:.3f} (negative space)"
            else:
                warp_str = f"{warp_factor:.3f}"
            
            self.status_var.set(f"✅ Galileo 3D: {len(concept_data)} concepts, Warp Factor: {warp_str}")
            
            # Update plot title
            self.ax.set_title(f'🚀 Interactive Galileo 3D Space | Warp Factor: {warp_str}', 
                             fontsize=14, fontweight='bold', pad=20)
            
            # Connect mouse events after plot is created
            if self.motion_event is None:
                self.motion_event = self.canvas.mpl_connect('motion_notify_event', self.on_hover)
            if self.click_event is None:
                self.click_event = self.canvas.mpl_connect('button_press_event', self.on_click)
            
            # Connect zoom and pan events
            if not hasattr(self, 'scroll_event'):
                self.scroll_event = self.canvas.mpl_connect('scroll_event', self.on_scroll)
            if not hasattr(self, 'drag_event'):
                self.drag_event = self.canvas.mpl_connect('motion_notify_event', self.on_drag)
            if not hasattr(self, 'press_event'):
                self.press_event = self.canvas.mpl_connect('button_press_event', self.on_press)
            if not hasattr(self, 'release_event'):
                self.release_event = self.canvas.mpl_connect('button_release_event', self.on_release)
            
            # Store original axis limits for reset functionality
            if self.original_xlim is None:
                self.original_xlim = self.ax.get_xlim()
                self.original_ylim = self.ax.get_ylim()
                self.original_zlim = self.ax.get_zlim()
            
            # Refresh canvas
            self.canvas.draw()
            
        except Exception as e:
            self.status_var.set(f"❌ Visualization error: {str(e)}")
            print(f"Interactive visualization error: {e}")
    
    def create_inline_sophisticated_3d_plot(self, num_concepts=50, min_mass=0.001):
        """
        Inline version of sophisticated 3D visualization for interactive window.
        """
        import numpy as np
        
        try:
            # Same algorithm as main GUI
            system_artifacts = {
                'speaker_speaker_a', 'speaker_speaker_b', 'speaker_speaker_c', 
                'speaker_speaker_d', 'speaker_speaker_e', 'speaker_speaker_f',
                're', 'e', 'g', '4', 'lines'
            }
            
            all_concepts = []
            
            # Check if this is plain text mode
            plain_text_mode = False
            if hasattr(self.huey, 'speakers') and len(self.huey.speakers) == 1:
                speaker_names = [s.lower() for s in self.huey.speakers.keys()]
                if 'text' in speaker_names:
                    plain_text_mode = True
            
            # Get speaker masses first (if available)
            if hasattr(self.huey, 'speakers') and self.huey.speakers and not plain_text_mode:
                for speaker in self.huey.speakers:
                    if hasattr(self.huey, 'analyze_speaker_self_concept'):
                        analysis = self.huey.analyze_speaker_self_concept(speaker)
                        mass = analysis.get('self_concept_mass', 0.0)
                        if mass > 0:
                            speaker_neuron_word = speaker.lower()
                            actual_speaker_id = self.huey.word_to_neuron.get(speaker_neuron_word)
                            if actual_speaker_id is not None:
                                all_concepts.append({
                                    'name': f"Speaker_{speaker}",
                                    'mass': mass,
                                    'id': actual_speaker_id
                                })
            
            # Get regular concept masses
            for neuron_id, word in self.huey.neuron_to_word.items():
                if not plain_text_mode and word.lower() in system_artifacts:
                    continue
                if plain_text_mode and word.lower() in {'speaker_text', 're', 'e', 'g', '4', 'lines'}:
                    continue
                    
                total_mass = 0.0
                if hasattr(self.huey, 'inertial_mass'):
                    for (i, j), mass in self.huey.inertial_mass.items():
                        if i == neuron_id or j == neuron_id:
                            total_mass += mass
                
                if total_mass >= min_mass:
                    all_concepts.append({
                        'name': word,
                        'mass': total_mass,
                        'id': neuron_id
                    })
            
            all_concepts.sort(key=lambda x: x['mass'], reverse=True)
            concepts_data = all_concepts[:num_concepts]
            
            if len(concepts_data) < 3:
                return None
            
            # Build association matrix
            n = len(concepts_data)
            concept_ids = [c['id'] for c in concepts_data]
            association_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        nid_i, nid_j = concept_ids[i], concept_ids[j]
                        conn_key1 = (nid_i, nid_j)
                        conn_key2 = (nid_j, nid_i)
                        
                        mass = 0.0
                        if hasattr(self.huey, 'inertial_mass'):
                            if conn_key1 in self.huey.inertial_mass:
                                mass += self.huey.inertial_mass[conn_key1]
                            if conn_key2 in self.huey.inertial_mass:
                                mass += self.huey.inertial_mass[conn_key2]
                        
                        association_matrix[i, j] = mass
            
            # Torgerson double-centering
            max_sim = np.max(association_matrix) if np.max(association_matrix) > 0 else 1.0
            distance_matrix = max_sim - association_matrix
            
            H = np.eye(n) - np.ones((n, n)) / n
            B = -0.5 * H @ (distance_matrix**2) @ H
            
            eigenvalues, eigenvectors = np.linalg.eigh(B)
            idx = np.argsort(eigenvalues)[::-1]
            
            selected_eigenvals = eigenvalues[idx[:3]]
            selected_eigenvecs = eigenvectors[:, idx[:3]]
            
            # Pseudo-Riemannian coordinate calculation
            positions = np.zeros((n, 3))
            for i in range(3):
                eigenval = selected_eigenvals[i]
                eigenvec = selected_eigenvecs[:, i]
                if eigenval > 1e-8:
                    positions[:, i] = eigenvec * np.sqrt(eigenval)
                elif eigenval < -1e-8:
                    positions[:, i] = eigenvec * np.sqrt(-eigenval)
                else:
                    positions[:, i] = eigenvec * 1e-4
            
            positions *= 3
            
            # Calculate warp factor
            positive_eigenvals = eigenvalues[eigenvalues > 1e-10]
            negative_eigenvals = eigenvalues[eigenvalues < -1e-10]
            
            if len(positive_eigenvals) > 0:
                sum_positive = np.sum(positive_eigenvals)
                sum_negative = np.sum(negative_eigenvals)
                sum_all = sum_positive + sum_negative
                
                if abs(sum_all) > 1e-8:
                    warp_factor = sum_positive / sum_all
                else:
                    warp_factor = float('inf')
            else:
                warp_factor = 1.0
            
            return positions, concepts_data, warp_factor
            
        except Exception as e:
            print(f"Error in inline sophisticated 3D plot: {e}")
            return None
    
    def calculate_label_positions(self, coords, labels):
        """Calculate intelligent label positions to minimize overlaps."""
        positions = []
        min_distance = 0.2  # Minimum distance between labels
        max_offset = 0.3    # Maximum offset from original position
        
        for i, coord in enumerate(coords):
            # Start with original position plus small offset
            pos = coord.copy()
            pos[2] += 0.08  # Slight z-offset for better visibility
            
            # Try multiple positions to find the best one
            best_pos = pos.copy()
            best_score = float('inf')
            
            # Test different offset directions
            offset_directions = [
                np.array([0, 0, 0.1]),      # Above
                np.array([0.1, 0, 0.05]),   # Right-up
                np.array([-0.1, 0, 0.05]),  # Left-up  
                np.array([0, 0.1, 0.05]),   # Back-up
                np.array([0, -0.1, 0.05]),  # Front-up
                np.array([0.15, 0, 0]),     # Right
                np.array([-0.15, 0, 0]),    # Left
            ]
            
            for offset in offset_directions:
                test_pos = coord + offset
                
                # Calculate overlap penalty with existing labels
                overlap_penalty = 0
                for other_pos in positions:
                    distance = np.linalg.norm(test_pos - other_pos)
                    if distance < min_distance:
                        overlap_penalty += (min_distance - distance) ** 2
                
                # Prefer positions closer to original
                distance_penalty = np.linalg.norm(offset) * 0.1
                
                total_score = overlap_penalty + distance_penalty
                
                if total_score < best_score:
                    best_score = total_score
                    best_pos = test_pos
            
            positions.append(best_pos)
        
        return positions
    
    def draw_concept_connections(self, coords, labels):
        """Draw connections between related concepts as lines."""
        if not hasattr(self.huey, 'connections') or len(self.huey.connections) == 0:
            return
        
        # Create concept to index mapping
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        
        # Track drawn connections to avoid duplicates
        drawn_connections = set()
        connection_count = 0
        max_connections = 50  # Limit to avoid visual clutter
        
        # Sort connections by strength for better visualization
        connection_strengths = []
        for conn_key, strength in self.huey.connections.items():
            if len(conn_key) == 2:  # Valid connection tuple
                neuron_a, neuron_b = conn_key
                
                # Find corresponding concept labels
                label_a = label_b = None
                for label in labels:
                    if label in self.huey.concept_neurons:
                        if self.huey.concept_neurons[label] == neuron_a:
                            label_a = label
                        elif self.huey.concept_neurons[label] == neuron_b:
                            label_b = label
                
                # If both concepts are in our visualization
                if label_a and label_b and label_a in label_to_idx and label_b in label_to_idx:
                    connection_strengths.append((strength, label_a, label_b))
        
        # Sort by strength (strongest first)
        connection_strengths.sort(reverse=True)
        
        # Draw top connections
        for strength, label_a, label_b in connection_strengths[:max_connections]:
            idx_a = label_to_idx[label_a]
            idx_b = label_to_idx[label_b]
            
            # Avoid duplicate visualization
            connection_pair = tuple(sorted([label_a, label_b]))
            if connection_pair not in drawn_connections:
                drawn_connections.add(connection_pair)
                
                # Get coordinates
                coord_a = coords[idx_a]
                coord_b = coords[idx_b]
                
                # Draw connection line with alpha based on strength
                alpha = min(0.8, max(0.1, strength / 2.0))  # Scale alpha by strength
                linewidth = max(0.5, min(3.0, strength * 2))  # Scale line width
                
                self.ax.plot([coord_a[0], coord_b[0]], 
                           [coord_a[1], coord_b[1]], 
                           [coord_a[2], coord_b[2]], 
                           color='gray', alpha=alpha, linewidth=linewidth,
                           linestyle='-' if strength > 1.0 else '--')
                
                connection_count += 1
        
        print(f"🔗 Drew {connection_count} connections between concepts")
    
    def update_concept_dropdown(self, labels):
        """Update the concept dropdown with available concepts."""
        # Sort concepts by mass for better UX
        concept_options = ["None"]  # Default option
        
        if hasattr(self, 'concept_masses') and len(labels) > 0:
            # Create list of (label, mass) pairs and sort by mass
            labeled_masses = list(zip(labels, self.concept_masses))
            labeled_masses.sort(key=lambda x: x[1], reverse=True)  # Sort by mass descending
            
            concept_options.extend([label for label, _ in labeled_masses])
        else:
            concept_options.extend(sorted(labels))
        
        # Update dropdown values
        self.concept_dropdown['values'] = concept_options
        
        # Reset selection if current selection is not available
        if self.selected_concept.get() not in concept_options:
            self.selected_concept.set("None")
    
    def on_concept_selected(self, event=None):
        """Handle concept selection from dropdown."""
        selected = self.selected_concept.get()
        
        if selected == "None":
            self.clear_concept_highlight()
            self.update_concept_details(None)
        else:
            self.highlight_concept(selected)
            self.update_concept_details(selected)
    
    def highlight_concept(self, concept_name):
        """Highlight a specific concept in the visualization."""
        if not self.labels_data or concept_name not in self.labels_data:
            return
        
        # Find the concept index
        concept_idx = None
        for i, label in enumerate(self.labels_data):
            if label == concept_name:
                concept_idx = i
                break
        
        if concept_idx is None:
            return
        
        # Clear previous highlights and redraw
        self.refresh_interactive_plot_with_highlight(concept_idx)
        
        print(f"🎯 Highlighted concept: {concept_name}")
    
    def refresh_interactive_plot_with_highlight(self, highlight_idx):
        """Refresh plot with specific concept highlighted."""
        if not self.huey or not hasattr(self.huey, 'concept_neurons'):
            return
            
        try:
            # Get coordinates (reuse existing data to maintain performance)
            coords_subset = self.coords_data
            labels_subset = self.labels_data
            
            if coords_subset is None or labels_subset is None:
                self.refresh_interactive_plot()  # Fallback to full refresh
                return
            
            # Clear and recreate plot
            self.create_interactive_plot()
            
            # Recreate scatter plot with highlighting
            sizes = [max(50, min(200, 50 + mass * 100)) for mass in self.concept_masses]
            colors = []
            alphas = []
            
            for i in range(len(labels_subset)):
                if i == highlight_idx:
                    colors.append('red')  # Highlight color
                    alphas.append(1.0)    # Full opacity
                    sizes[i] = max(sizes[i], 250)  # Make highlighted concept larger
                else:
                    colors.append('blue')  # Default color
                    alphas.append(0.6)    # Reduced opacity
            
            # Create scatter plot with highlighting
            self.scatter_plot = self.ax.scatter(coords_subset[:, 0], coords_subset[:, 1], coords_subset[:, 2],
                                    c=colors, alpha=0.7, s=sizes, edgecolors='black', linewidth=1.5)
            
            # Add labels (highlight the selected one)
            label_positions = self.calculate_label_positions(coords_subset, labels_subset)
            
            for i, (label, pos) in enumerate(zip(labels_subset, label_positions)):
                if i == highlight_idx:
                    # Highlighted concept gets special treatment
                    self.ax.text(pos[0], pos[1], pos[2], 
                               f"🎯 {label}", fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", 
                                       facecolor='yellow', alpha=0.9, edgecolor='red', linewidth=2),
                               ha='center', va='bottom')
                elif i < 15 or (hasattr(self, 'concept_masses') and self.concept_masses[i] > np.mean(self.concept_masses)):
                    # Other important concepts
                    self.ax.text(pos[0], pos[1], pos[2], 
                               label, fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", 
                                       facecolor='white', alpha=0.8, edgecolor='gray'),
                               ha='center', va='bottom')
            
            # Draw connections (dimmed except those involving highlighted concept)
            self.draw_concept_connections_with_highlight(coords_subset, labels_subset, highlight_idx)
            
            # Refresh canvas
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error in highlight refresh: {e}")
            self.refresh_interactive_plot()  # Fallback
    
    def draw_concept_connections_with_highlight(self, coords, labels, highlight_idx):
        """Draw connections with emphasis on highlighted concept."""
        if not hasattr(self.huey, 'connections') or len(self.huey.connections) == 0:
            return
        
        highlighted_label = labels[highlight_idx] if highlight_idx < len(labels) else None
        highlighted_neuron = None
        
        if highlighted_label and highlighted_label in self.huey.concept_neurons:
            highlighted_neuron = self.huey.concept_neurons[highlighted_label]
        
        # Create concept to index mapping
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        
        # Draw connections with special treatment for highlighted concept
        connection_count = 0
        max_connections = 50
        
        # Sort connections by relevance to highlighted concept
        connection_strengths = []
        for conn_key, strength in self.huey.connections.items():
            if len(conn_key) == 2:
                neuron_a, neuron_b = conn_key
                
                # Find corresponding concept labels
                label_a = label_b = None
                for label in labels:
                    if label in self.huey.concept_neurons:
                        if self.huey.concept_neurons[label] == neuron_a:
                            label_a = label
                        elif self.huey.concept_neurons[label] == neuron_b:
                            label_b = label
                
                if label_a and label_b and label_a in label_to_idx and label_b in label_to_idx:
                    # Prioritize connections involving highlighted concept
                    priority = 0
                    if highlighted_neuron and (neuron_a == highlighted_neuron or neuron_b == highlighted_neuron):
                        priority = 10  # High priority for highlighted connections
                    
                    connection_strengths.append((strength + priority, label_a, label_b, 
                                              neuron_a == highlighted_neuron or neuron_b == highlighted_neuron))
        
        # Sort by strength (with priority boost for highlighted connections)
        connection_strengths.sort(reverse=True)
        
        # Draw connections
        drawn_connections = set()
        for strength, label_a, label_b, involves_highlight in connection_strengths[:max_connections]:
            idx_a = label_to_idx[label_a]
            idx_b = label_to_idx[label_b]
            
            connection_pair = tuple(sorted([label_a, label_b]))
            if connection_pair not in drawn_connections:
                drawn_connections.add(connection_pair)
                
                coord_a = coords[idx_a]
                coord_b = coords[idx_b]
                
                # Special styling for highlighted connections
                if involves_highlight:
                    color = 'red'
                    alpha = 0.9
                    linewidth = max(2.0, min(5.0, (strength - 10) * 3))  # Subtract priority boost
                    linestyle = '-'
                else:
                    color = 'gray'
                    alpha = min(0.4, max(0.1, strength / 2.0))
                    linewidth = max(0.5, min(2.0, strength * 1.5))
                    linestyle = '--'
                
                self.ax.plot([coord_a[0], coord_b[0]], 
                           [coord_a[1], coord_b[1]], 
                           [coord_a[2], coord_b[2]], 
                           color=color, alpha=alpha, linewidth=linewidth,
                           linestyle=linestyle)
                
                connection_count += 1
    
    def clear_concept_highlight(self):
        """Clear concept highlighting and return to normal view."""
        self.refresh_interactive_plot()
        print("🔄 Cleared concept highlighting")
    
    def update_concept_details(self, concept_name):
        """Update the concept details panel with information about selected concept."""
        self.concept_details.config(state=tk.NORMAL)
        self.concept_details.delete(1.0, tk.END)
        
        if concept_name is None or concept_name == "None":
            self.concept_details.insert(tk.END, "No concept selected. Choose a concept from the dropdown to see detailed analysis.")
        else:
            # Get detailed information about the concept
            details = self.get_concept_cluster_analysis(concept_name)
            self.concept_details.insert(tk.END, details)
        
        self.concept_details.config(state=tk.DISABLED)
    
    def get_concept_cluster_analysis(self, concept_name):
        """Generate detailed cluster analysis for a specific concept."""
        if concept_name not in self.huey.concept_neurons:
            return f"Concept '{concept_name}' not found in network."
        
        neuron_id = self.huey.concept_neurons[concept_name]
        
        # Get basic metrics
        activation = self.huey.activations.get(neuron_id, 0.0)
        
        # Calculate mass based on activation and connections (same formula as visualization)
        conn_count_for_mass = sum(1 for key in self.huey.connections.keys() if neuron_id in key)
        mass = activation * 0.5 + conn_count_for_mass * 0.1
        
        # Find connections
        connected_concepts = []
        connection_strengths = []
        
        for conn_key, strength in self.huey.connections.items():
            if len(conn_key) == 2 and neuron_id in conn_key:
                other_neuron = conn_key[0] if conn_key[1] == neuron_id else conn_key[1]
                
                # Find the concept name for the other neuron
                other_concept = None
                for concept, n_id in self.huey.concept_neurons.items():
                    if n_id == other_neuron:
                        other_concept = concept
                        break
                
                if other_concept:
                    connected_concepts.append(other_concept)
                    connection_strengths.append(strength)
        
        # Sort connections by strength
        if connected_concepts:
            connected_pairs = list(zip(connected_concepts, connection_strengths))
            connected_pairs.sort(key=lambda x: x[1], reverse=True)
            top_connections = connected_pairs[:8]  # Show top 8 connections
        else:
            top_connections = []
        
        # Get spatial coordinates
        coords, _, labels, _ = self.huey.get_3d_coordinates()
        concept_coords = None
        if concept_name in labels:
            idx = labels.index(concept_name)
            concept_coords = coords[idx] if idx < len(coords) else None
        
        # Build analysis text
        analysis = f"""🎯 CONCEPT CLUSTER ANALYSIS: {concept_name.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Network Metrics:
   • Inertial Mass: {mass:.4f}
   • Current Activation: {activation:.4f}  
   • Total Connections: {len(connected_concepts)}
   • Network Position: {concept_coords if concept_coords is not None else 'Not available'}

🔗 Top Connected Concepts ({len(top_connections)} of {len(connected_concepts)}):"""
        
        for i, (connected_concept, strength) in enumerate(top_connections):
            analysis += f"\n   {i+1:2d}. {connected_concept:<15} (strength: {strength:.3f})"
        
        if not top_connections:
            analysis += "\n   No strong connections found."
        
        analysis += f"""
        
💡 Cluster Analysis:
   • Centrality: {'High' if len(connected_concepts) > 5 else 'Medium' if len(connected_concepts) > 2 else 'Low'}
   • Influence: {'Strong' if mass > 1.0 else 'Moderate' if mass > 0.5 else 'Weak'}
   • Activity: {'Active' if activation > 0.5 else 'Moderate' if activation > 0.1 else 'Dormant'}"""
        
        return analysis
    
    def on_hover(self, event):
        """Handle mouse hover events for tooltips."""
        if event.inaxes != self.ax or self.scatter_plot is None:
            return
        
        # Find nearest point using 3D distance in data coordinates
        if hasattr(event, 'xdata') and event.xdata is not None and hasattr(event, 'ydata'):
            if self.coords_data is not None and len(self.coords_data) > 0:
                # Simple 3D distance calculation in data coordinates
                mouse_3d = np.array([event.xdata, event.ydata, 0])  # Z=0 for mouse position
                distances = []
                
                for i, coord in enumerate(self.coords_data):
                    # Calculate distance in 3D data space (ignoring Z for mouse)
                    dist = np.sqrt((coord[0] - event.xdata)**2 + (coord[1] - event.ydata)**2)
                    distances.append(dist)
                
                # Find closest point
                closest_idx = np.argmin(distances)
                
                # Show tooltip if close enough (data coordinate threshold)
                # Use adaptive threshold based on data scale
                data_range = np.max(self.coords_data) - np.min(self.coords_data) if len(self.coords_data) > 0 else 1.0
                threshold = max(0.1, data_range * 0.05)  # 5% of data range, minimum 0.1
                
                if distances[closest_idx] < threshold:
                    self.show_tooltip(closest_idx, event)
                else:
                    self.hide_tooltip()
    
    def show_tooltip(self, point_idx, event):
        """Display tooltip with concept information."""
        if point_idx >= len(self.labels_data):
            return
        
        label = self.labels_data[point_idx]
        mass = self.concept_masses[point_idx] if hasattr(self, 'concept_masses') else 0.0
        connections = self.connection_counts[point_idx] if hasattr(self, 'connection_counts') else 0
        
        # Get activation level if available
        activation = 0.0
        if label in self.huey.concept_neurons:
            neuron_id = self.huey.concept_neurons[label]
            activation = self.huey.activations.get(neuron_id, 0.0)
        
        # Create tooltip text
        tooltip_info = f"""🧠 {label}
📊 Mass: {mass:.3f}
⚡ Activation: {activation:.3f}
🔗 Connections: {connections}
📍 Coords: ({self.coords_data[point_idx][0]:.2f}, {self.coords_data[point_idx][1]:.2f}, {self.coords_data[point_idx][2]:.2f})"""
        
        # Update status with hover info
        self.status_var.set(f"🎯 Hovering: {label} (Mass: {mass:.3f}, Connections: {connections})")
    
    def hide_tooltip(self):
        """Hide the tooltip."""
        # Reset status to default
        if hasattr(self, 'concept_masses') and self.coords_data is not None:
            total_concepts = len(self.coords_data)
            total_connections = len(getattr(self.huey, 'connections', {}))
            self.status_var.set(f"✅ Showing {total_concepts} concepts, {total_connections} connections")
    
    def on_click(self, event):
        """Handle mouse click events for concept selection."""
        if event.inaxes != self.ax or self.scatter_plot is None:
            return
        
        # Find clicked point (similar to hover logic)
        if hasattr(event, 'xdata') and event.xdata is not None and hasattr(event, 'ydata') and self.coords_data is not None:
            distances = []
            
            for i, coord in enumerate(self.coords_data):
                # Calculate distance in 3D data space (ignoring Z for mouse)
                dist = np.sqrt((coord[0] - event.xdata)**2 + (coord[1] - event.ydata)**2)
                distances.append(dist)
            
            closest_idx = np.argmin(distances)
            
            # Use adaptive threshold (same as hover)
            data_range = np.max(self.coords_data) - np.min(self.coords_data) if len(self.coords_data) > 0 else 1.0
            threshold = max(0.1, data_range * 0.05)  # 5% of data range, minimum 0.1
            
            if distances[closest_idx] < threshold:
                self.select_concept(closest_idx)
    
    def select_concept(self, point_idx):
        """Select a concept and highlight it."""
        if point_idx >= len(self.labels_data):
            return
        
        label = self.labels_data[point_idx]
        
        # Highlight selected concept
        self.selected_point = point_idx
        
        # Update status with selection info
        mass = self.concept_masses[point_idx] if hasattr(self, 'concept_masses') else 0.0
        connections = self.connection_counts[point_idx] if hasattr(self, 'connection_counts') else 0
        
        self.status_var.set(f"🎯 Selected: {label} (Mass: {mass:.3f}, {connections} connections)")
        
        print(f"🎯 Concept selected: {label} (Mass: {mass:.3f}, Connections: {connections})")
    
    def on_scroll(self, event):
        """Handle mouse scroll events for zooming."""
        if event.inaxes != self.ax:
            return
        
        # Determine zoom direction
        zoom_in = event.step > 0
        zoom_factor = 1.2 if zoom_in else 1/1.2
        
        # Get current axis limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()
        
        # Calculate zoom center (mouse position or plot center)
        if hasattr(event, 'xdata') and event.xdata is not None:
            center_x, center_y = event.xdata, event.ydata
        else:
            center_x = (xlim[0] + xlim[1]) / 2
            center_y = (ylim[0] + ylim[1]) / 2
        
        # Calculate center for Z (middle of current Z range)
        center_z = (zlim[0] + zlim[1]) / 2
        
        # Calculate new limits
        x_range = (xlim[1] - xlim[0]) / zoom_factor
        y_range = (ylim[1] - ylim[0]) / zoom_factor
        z_range = (zlim[1] - zlim[0]) / zoom_factor
        
        new_xlim = [center_x - x_range/2, center_x + x_range/2]
        new_ylim = [center_y - y_range/2, center_y + y_range/2]
        new_zlim = [center_z - z_range/2, center_z + z_range/2]
        
        # Apply new limits
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.ax.set_zlim(new_zlim)
        
        # Update zoom level for status
        self.zoom_level *= zoom_factor
        
        # Update canvas
        self.canvas.draw()
        
        # Update status
        zoom_percent = int(self.zoom_level * 100)
        self.status_var.set(f"🔍 Zoom: {zoom_percent}% | Scroll to zoom, drag to pan")
    
    def on_press(self, event):
        """Handle mouse button press for panning."""
        if event.inaxes != self.ax:
            return
            
        if event.button == 2:  # Middle mouse button for panning
            self.pan_start = (event.xdata, event.ydata)
            self.is_panning = True
            self.status_var.set("🤚 Panning - release middle mouse to stop")
        elif event.button == 1:  # Left click - let normal selection handle it
            # Don't interfere with concept selection
            pass
    
    def on_release(self, event):
        """Handle mouse button release."""
        if event.button == 2:  # Middle mouse button
            self.is_panning = False
            self.pan_start = None
            # Restore normal status
            concept_count = len(self.labels_data) if self.labels_data else 0
            connection_count = len(getattr(self.huey, 'connections', {}))
            self.status_var.set(f"✅ Showing {concept_count} concepts, {connection_count} connections")
    
    def on_drag(self, event):
        """Handle mouse drag events for panning."""
        if not self.is_panning or self.pan_start is None or event.inaxes != self.ax:
            return
        
        if hasattr(event, 'xdata') and event.xdata is not None:
            # Calculate pan delta
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            
            # Get current limits
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            # Apply pan (move in opposite direction of drag)
            new_xlim = [xlim[0] - dx, xlim[1] - dx]
            new_ylim = [ylim[0] - dy, ylim[1] - dy]
            
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            
            # Update pan start position for continuous dragging
            self.pan_start = (event.xdata, event.ydata)
            
            # Update canvas
            self.canvas.draw()
    
    def reset_view(self):
        """Reset the 3D view to default."""
        self.ax.view_init(elev=20, azim=45)
        
        # Reset zoom and pan to original limits
        if self.original_xlim and self.original_ylim and self.original_zlim:
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.ax.set_zlim(self.original_zlim)
            self.zoom_level = 1.0
        
        self.canvas.draw()
        self.status_var.set("📍 View, zoom, and pan reset to default")
    
    def save_image(self):
        """Save the current visualization as an image."""
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG files', '*.png'), ('PDF files', '*.pdf'), ('SVG files', '*.svg')],
            title="Save Interactive Visualization"
        )
        
        if filename:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
            self.status_var.set(f"💾 Saved: {os.path.basename(filename)}")
def main():
    """Main entry point."""
    app = HueyGUIGPU()
    app.run()

if __name__ == "__main__":
    main()