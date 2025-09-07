#!/usr/bin/env python3
"""
Test PyOpenGLTK for 3D Interactive Visualization with Tkinter
"""

import tkinter as tk
import numpy as np
from pyopengltk import OpenGLFrame
from OpenGL import GL
import math

class Galileo3DFrame(OpenGLFrame):
    """3D visualization frame using OpenGL for Galileo concept space"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # 3D visualization parameters
        self.rotation_x = 0
        self.rotation_y = 0
        self.zoom = -5.0
        self.mouse_down = False
        self.last_x = 0
        self.last_y = 0
        
        # Generate test concept data
        self.generate_concept_data()
        
        # Bind mouse events for interaction
        self.bind('<Button-1>', self.on_mouse_down)
        self.bind('<B1-Motion>', self.on_mouse_drag)
        self.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.bind('<MouseWheel>', self.on_mouse_wheel)
        
    def generate_concept_data(self):
        """Generate test data similar to Galileo concept space"""
        n_concepts = 50
        np.random.seed(42)
        
        # Simulate pseudo-Riemannian embedding results
        self.positions = np.random.randn(n_concepts, 3) * 2
        self.masses = np.random.rand(n_concepts) * 10 + 1
        self.colors = []
        
        # Generate colors based on masses
        for mass in self.masses:
            # Plasma colormap approximation
            normalized_mass = (mass - np.min(self.masses)) / (np.max(self.masses) - np.min(self.masses))
            r = min(1.0, normalized_mass * 1.5)
            g = normalized_mass * 0.8
            b = min(1.0, (1 - normalized_mass) * 2)
            self.colors.append([r, g, b])
        
        self.labels = [f"concept_{i}" for i in range(n_concepts)]
        
    def initgl(self):
        """Initialize OpenGL settings"""
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)  # Dark background
        GL.glEnable(GL.GL_POINT_SMOOTH)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        
        # Set up perspective projection
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        self.perspective(45, self.width/self.height, 0.1, 100.0)
        
    def perspective(self, fovy, aspect, znear, zfar):
        """Set perspective projection"""
        ymax = znear * math.tan(fovy * math.pi / 360.0)
        ymin = -ymax
        xmin = ymin * aspect
        xmax = ymax * aspect
        GL.glFrustum(xmin, xmax, ymin, ymax, znear, zfar)
        
    def redraw(self):
        """Redraw the 3D scene"""
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        
        # Apply transformations
        GL.glTranslatef(0.0, 0.0, self.zoom)
        GL.glRotatef(self.rotation_x, 1, 0, 0)
        GL.glRotatef(self.rotation_y, 0, 1, 0)
        
        # Draw coordinate axes
        self.draw_axes()
        
        # Draw concept points
        self.draw_concepts()
        
        # Draw some connections between nearby concepts
        self.draw_connections()
        
    def draw_axes(self):
        """Draw 3D coordinate axes"""
        GL.glBegin(GL.GL_LINES)
        
        # X axis (red)
        GL.glColor3f(1.0, 0.0, 0.0)
        GL.glVertex3f(0.0, 0.0, 0.0)
        GL.glVertex3f(3.0, 0.0, 0.0)
        
        # Y axis (green)  
        GL.glColor3f(0.0, 1.0, 0.0)
        GL.glVertex3f(0.0, 0.0, 0.0)
        GL.glVertex3f(0.0, 3.0, 0.0)
        
        # Z axis (blue)
        GL.glColor3f(0.0, 0.0, 1.0)
        GL.glVertex3f(0.0, 0.0, 0.0)
        GL.glVertex3f(0.0, 0.0, 3.0)
        
        GL.glEnd()
        
    def draw_concepts(self):
        """Draw concept points as spheres with mass-based sizing"""
        for i, (pos, mass, color) in enumerate(zip(self.positions, self.masses, self.colors)):
            GL.glPushMatrix()
            GL.glTranslatef(pos[0], pos[1], pos[2])
            
            # Color based on mass
            GL.glColor3f(*color)
            
            # Size based on mass
            size = 0.1 + (mass / 10.0) * 0.3
            
            # Draw simple sphere (approximated as points for now)
            GL.glPointSize(size * 20)
            GL.glBegin(GL.GL_POINTS)
            GL.glVertex3f(0, 0, 0)
            GL.glEnd()
            
            GL.glPopMatrix()
            
    def draw_connections(self):
        """Draw connections between nearby concepts"""
        GL.glColor4f(0.5, 0.5, 0.5, 0.3)  # Semi-transparent gray
        GL.glBegin(GL.GL_LINES)
        
        # Draw connections between concepts within distance threshold
        threshold = 2.0
        for i, pos1 in enumerate(self.positions):
            for j, pos2 in enumerate(self.positions[i+1:], i+1):
                distance = np.linalg.norm(pos1 - pos2)
                if distance < threshold:
                    GL.glVertex3f(*pos1)
                    GL.glVertex3f(*pos2)
        
        GL.glEnd()
        
    def on_mouse_down(self, event):
        """Handle mouse down for rotation"""
        self.mouse_down = True
        self.last_x = event.x
        self.last_y = event.y
        
    def on_mouse_drag(self, event):
        """Handle mouse drag for rotation"""
        if self.mouse_down:
            dx = event.x - self.last_x
            dy = event.y - self.last_y
            
            self.rotation_y += dx * 0.5
            self.rotation_x += dy * 0.5
            
            self.last_x = event.x
            self.last_y = event.y
            
            self.tkRedraw()
            
    def on_mouse_up(self, event):
        """Handle mouse up"""
        self.mouse_down = False
        
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming"""
        if event.delta > 0:
            self.zoom += 0.5
        else:
            self.zoom -= 0.5
            
        # Limit zoom range
        self.zoom = max(-20.0, min(-1.0, self.zoom))
        self.tkRedraw()

def create_test_app():
    """Create test application with PyOpenGLTK 3D visualization"""
    
    root = tk.Tk()
    root.title("🚀 PyOpenGLTK Galileo 3D Concept Space Test")
    root.geometry("900x700")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Control frame
    control_frame = tk.Frame(main_frame, height=60, bg='lightgray')
    control_frame.pack(fill=tk.X, padx=5, pady=5)
    control_frame.pack_propagate(False)
    
    # 3D visualization frame
    viz_frame = tk.Frame(main_frame, relief=tk.SUNKEN, bd=2)
    viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Create OpenGL 3D widget
    gl_frame = Galileo3DFrame(viz_frame, width=850, height=600)
    gl_frame.pack(fill=tk.BOTH, expand=True)
    gl_frame.animate = 1  # Enable continuous redraw for smooth interaction
    
    # Control buttons
    def regenerate_data():
        """Generate new concept data"""
        gl_frame.generate_concept_data()
        gl_frame.tkRedraw()
        
    def reset_view():
        """Reset view to default"""
        gl_frame.rotation_x = 0
        gl_frame.rotation_y = 0
        gl_frame.zoom = -5.0
        gl_frame.tkRedraw()
    
    # Add controls
    tk.Label(control_frame, text="🚀 PyOpenGLTK Galileo 3D Concept Space", 
             font=('Arial', 14, 'bold'), bg='lightgray').pack(side=tk.LEFT)
    
    tk.Button(control_frame, text="🔄 New Data", command=regenerate_data, 
              font=('Arial', 10)).pack(side=tk.LEFT, padx=10)
    
    tk.Button(control_frame, text="🎯 Reset View", command=reset_view,
              font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
    
    tk.Button(control_frame, text="❌ Close", command=root.quit,
              font=('Arial', 10)).pack(side=tk.RIGHT, padx=10)
    
    # Instructions
    instructions = tk.Label(control_frame, 
                           text="Mouse: Drag to rotate | Wheel: Zoom | Interactive 3D OpenGL", 
                           font=('Arial', 9), bg='lightgray', fg='blue')
    instructions.pack(side=tk.RIGHT, padx=20)
    
    print("🚀 PyOpenGLTK 3D test window opened!")
    print("   - Drag mouse to rotate the 3D view")
    print("   - Use mouse wheel to zoom in/out")  
    print("   - Click buttons to test interactivity")
    print("   - This uses native OpenGL for smooth performance!")
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    print("🧪 Testing PyOpenGLTK 3D Interactive Visualization...")
    create_test_app()
    print("✅ Test completed!")