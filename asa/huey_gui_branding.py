#!/usr/bin/env python3
"""
🎨 Huey GUI Branding Support
===========================

Image and branding support for Huey GUI GPU.
Handles loading, resizing, and displaying custom images for professional appearance.

Copyright (c) 2025 Emary Iacobucci and Joseph Woelfel. All rights reserved.
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
from typing import Optional, Tuple, Dict

class HueyBrandingManager:
    """Manages branding assets and visual elements for Huey GUI."""
    
    def __init__(self):
        """Initialize the branding manager."""
        self.images = {}
        self.logo_cache = {}
        
        # Galileo brand colors
        self.brand_colors = {
            'galileo_blue': '#2E4057',
            'galileo_teal': '#048A81', 
            'galileo_orange': '#F39C12',
            'galileo_green': '#27AE60',
            'galileo_red': '#E74C3C',
            'galileo_light': '#ECF0F1',
            'galileo_dark': '#2C3E50',
            'galileo_white': '#FFFFFF',
            'galileo_gold': '#D4AF37'
        }
    
    def load_image(self, image_path: str, size: Optional[Tuple[int, int]] = None) -> Optional[ImageTk.PhotoImage]:
        """
        Load and optionally resize an image for use in Tkinter.
        
        Args:
            image_path: Path to the image file
            size: Optional (width, height) tuple for resizing
            
        Returns:
            ImageTk.PhotoImage object or None if loading fails
        """
        try:
            if not os.path.exists(image_path):
                print(f"⚠️ Image not found: {image_path}")
                return self.create_placeholder_logo(size or (100, 100))
            
            # Load and process image
            img = Image.open(image_path)
            
            # Convert to RGBA for transparency support
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Resize if requested
            if size:
                img = img.resize(size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Cache the image
            cache_key = f"{image_path}_{size}"
            self.images[cache_key] = photo
            
            return photo
            
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return self.create_placeholder_logo(size or (100, 100))
    
    def create_placeholder_logo(self, size: Tuple[int, int]) -> ImageTk.PhotoImage:
        """
        Create a professional placeholder logo when image files aren't available.
        
        Args:
            size: (width, height) tuple for the logo
            
        Returns:
            ImageTk.PhotoImage with Galileo branding
        """
        width, height = size
        
        # Create new image with gradient background
        img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        # Create gradient background
        for y in range(height):
            # Interpolate between galileo_blue and galileo_teal
            ratio = y / height
            r = int(46 * (1 - ratio) + 4 * ratio)    # Blue to teal red component
            g = int(64 * (1 - ratio) + 138 * ratio)  # Blue to teal green component
            b = int(87 * (1 - ratio) + 129 * ratio)  # Blue to teal blue component
            
            draw.rectangle([(0, y), (width, y + 1)], fill=(r, g, b, 255))
        
        # Add border
        border_color = self.hex_to_rgb(self.brand_colors['galileo_dark'])
        draw.rectangle([(0, 0), (width-1, height-1)], outline=border_color, width=2)
        
        # Add text
        try:
            # Try to use a nice font
            font_size = min(width // 8, height // 4, 16)
            font = ImageFont.truetype("Arial", font_size)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Draw "GALILEO" text
        text = "GALILEO"
        text_color = self.hex_to_rgb(self.brand_colors['galileo_white'])
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center the text
        x = (width - text_width) // 2
        y = (height - text_height) // 2 - height // 8
        
        draw.text((x, y), text, font=font, fill=text_color)
        
        # Add subtitle
        subtitle = "HUEY"
        subtitle_size = max(font_size // 2, 8)
        try:
            subtitle_font = ImageFont.truetype("Arial", subtitle_size)
        except:
            subtitle_font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = (height - text_height) // 2 + height // 6
        
        draw.text((x, y), subtitle, font=subtitle_font, fill=text_color)
        
        return ImageTk.PhotoImage(img)
    
    def hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def create_branded_button_style(self, parent_widget, style_name: str = "Branded.TButton"):
        """
        Create a custom button style with Galileo branding.
        
        Args:
            parent_widget: Parent Tkinter widget to apply style to
            style_name: Name for the custom style
        """
        style = ttk.Style()
        
        # Configure the branded button style with dark text
        style.configure(style_name,
                       background=self.brand_colors['galileo_light'],
                       foreground=self.brand_colors['galileo_dark'],
                       font=('Arial', 10, 'bold'),
                       borderwidth=2,
                       relief='raised',
                       focuscolor=self.brand_colors['galileo_teal'])
        
        # Configure hover and state changes with dark text
        style.map(style_name,
                  background=[('active', self.brand_colors['galileo_white']),
                             ('pressed', '#D5DBDB')],  # Light gray when pressed
                  foreground=[('active', self.brand_colors['galileo_dark']),
                             ('pressed', self.brand_colors['galileo_dark']),
                             ('disabled', '#888888')])
        
        return style_name
    
    def setup_branded_window(self, root: tk.Tk):
        """
        Apply Galileo branding to the main window.
        
        Args:
            root: Main Tkinter window
        """
        # Set window icon (if available)
        try:
            icon_path = "galileo_icon.ico"
            if os.path.exists(icon_path):
                root.iconbitmap(icon_path)
        except:
            pass
        
        # Configure window background
        root.configure(bg=self.brand_colors['galileo_light'])
        
        # Apply branded ttk styles
        style = ttk.Style()
        
        # Configure frame styles
        style.configure('Branded.TFrame',
                       background=self.brand_colors['galileo_light'],
                       borderwidth=1,
                       relief='flat')
        
        # Configure label styles  
        style.configure('BrandedTitle.TLabel',
                       background=self.brand_colors['galileo_light'],
                       foreground=self.brand_colors['galileo_blue'],
                       font=('Arial', 16, 'bold'))
        
        style.configure('BrandedSubtitle.TLabel', 
                       background=self.brand_colors['galileo_light'],
                       foreground=self.brand_colors['galileo_teal'],
                       font=('Arial', 11))
        
        # Configure notebook styles with high contrast
        style.configure('Branded.TNotebook',
                       background=self.brand_colors['galileo_light'],
                       borderwidth=0)
        
        style.configure('Branded.TNotebook.Tab',
                       background=self.brand_colors['galileo_white'],
                       foreground=self.brand_colors['galileo_dark'],
                       padding=[20, 10],
                       font=('Arial', 10, 'bold'),
                       borderwidth=1)
        
        style.map('Branded.TNotebook.Tab',
                  background=[('selected', '#AED6F1'),  # Light blue for selected
                             ('active', '#D5F4E6')],      # Light green for hover
                  foreground=[('selected', self.brand_colors['galileo_dark']),
                             ('active', self.brand_colors['galileo_dark']),
                             ('!active', self.brand_colors['galileo_dark'])])
    
    def create_status_indicator(self, parent, status: str, size: Tuple[int, int] = (20, 20)) -> tk.Label:
        """
        Create a colored status indicator.
        
        Args:
            parent: Parent widget
            status: Status type ('success', 'warning', 'error', 'info')
            size: (width, height) of the indicator
            
        Returns:
            tk.Label with colored indicator
        """
        width, height = size
        
        # Color mapping
        status_colors = {
            'success': self.brand_colors['galileo_green'],
            'warning': self.brand_colors['galileo_orange'], 
            'error': self.brand_colors['galileo_red'],
            'info': self.brand_colors['galileo_blue']
        }
        
        color = status_colors.get(status, self.brand_colors['galileo_blue'])
        
        # Create circular indicator
        img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw circle
        margin = 2
        rgb_color = self.hex_to_rgb(color)
        draw.ellipse([(margin, margin), (width-margin, height-margin)], 
                    fill=rgb_color, outline=self.hex_to_rgb(self.brand_colors['galileo_dark']))
        
        photo = ImageTk.PhotoImage(img)
        
        # Create label
        label = tk.Label(parent, image=photo, background=self.brand_colors['galileo_light'])
        label.image = photo  # Keep a reference
        
        return label
    
    def load_banner_image(self, banner_path: str, canvas_width: int) -> Optional[ImageTk.PhotoImage]:
        """
        Load and prepare a banner image for the header.
        
        Args:
            banner_path: Path to banner image
            canvas_width: Width to fit the banner to
            
        Returns:
            Prepared PhotoImage or placeholder
        """
        if os.path.exists(banner_path):
            try:
                img = Image.open(banner_path)
                
                # Calculate height maintaining aspect ratio
                aspect_ratio = img.height / img.width
                new_height = int(canvas_width * aspect_ratio)
                
                # Ensure reasonable height limits
                new_height = min(max(new_height, 80), 150)
                
                img = img.resize((canvas_width, new_height), Image.Resampling.LANCZOS)
                
                return ImageTk.PhotoImage(img)
                
            except Exception as e:
                print(f"❌ Error loading banner: {e}")
                return None
        else:
            print(f"⚠️ Banner image not found: {banner_path}")
            return None

def create_image_structure():
    """
    Create recommended directory structure for branding images.
    Prints instructions for adding your Galileo branding assets.
    """
    print("🎨 GALILEO BRANDING SETUP")
    print("=" * 50)
    print()
    print("To add your Galileo branding to Huey GUI GPU, create these files:")
    print()
    print("📁 Image Files (place in same directory as huey_gui_gpu.py):")
    print("   • galileo_logo.png         - Main logo (recommended: 200x100px)")
    print("   • galileo_banner.png       - Header banner (recommended: 1200x150px)")
    print("   • galileo_icon.ico         - Window icon (recommended: 32x32px)")
    print("   • galileo_background.png   - Optional background image")
    print()
    print("🎯 Logo Requirements:")
    print("   • PNG format with transparency support")
    print("   • High resolution for crisp display")
    print("   • Consistent with Galileo brand guidelines")
    print()
    print("🔧 Integration Example:")
    print("   In your huey_gui_gpu.py, replace the placeholder logo creation with:")
    print("   logo_img = branding.load_image('galileo_logo.png', (200, 100))")
    print()
    print("✨ The branding manager will automatically create professional")
    print("   placeholders if image files are not found, so your app will")
    print("   work immediately and look great once you add your images!")

if __name__ == "__main__":
    create_image_structure()