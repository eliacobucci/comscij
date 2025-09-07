#!/usr/bin/env python3
"""
Test PyVista-Tkinter Integration for 3D Interactive Visualization
"""

import tkinter as tk
import pyvista as pv
from vtk.tk.vtkTkRenderWindowInteractor import vtkTkRenderWindowInteractor
import numpy as np

def create_test_3d_plot():
    """Create test 3D visualization with PyVista"""
    
    # Create root window
    root = tk.Tk()
    root.title("🚀 PyVista-Tkinter 3D Integration Test")
    root.geometry("800x600")
    
    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Create control frame
    control_frame = tk.Frame(main_frame, height=50)
    control_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Create visualization frame
    viz_frame = tk.Frame(main_frame)
    viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Create PyVista plotter
    plotter = pv.Plotter()
    
    # Generate test data similar to Galileo concept space
    n_points = 50
    np.random.seed(42)
    
    # Create 3D scatter data
    x = np.random.randn(n_points) * 2
    y = np.random.randn(n_points) * 2  
    z = np.random.randn(n_points) * 2
    
    # Create point cloud
    points = np.column_stack([x, y, z])
    point_cloud = pv.PolyData(points)
    
    # Add some concept masses as scalar data
    masses = np.random.rand(n_points) * 10
    point_cloud['masses'] = masses
    
    # Add mesh to plotter with mass-based coloring
    plotter.add_mesh(point_cloud, scalars='masses', 
                    point_size=15, render_points_as_spheres=True,
                    cmap='plasma', show_scalar_bar=True)
    
    # Add axes
    plotter.add_axes(color='black')
    
    # Set labels
    plotter.add_text("PyVista 3D Concept Space Test", position='upper_left', font_size=14)
    
    # Create VTK-Tkinter render window
    render_widget = vtkTkRenderWindowInteractor(viz_frame, 
                                                rw=plotter.ren_win,
                                                width=750, 
                                                height=500)
    render_widget.Initialize()
    render_widget.pack(fill=tk.BOTH, expand=True)
    render_widget.Start()
    
    # Add control buttons
    def refresh_data():
        """Refresh with new random data"""
        plotter.clear()
        
        # Generate new data
        new_x = np.random.randn(n_points) * 3
        new_y = np.random.randn(n_points) * 3
        new_z = np.random.randn(n_points) * 3
        new_points = np.column_stack([new_x, new_y, new_z])
        new_cloud = pv.PolyData(new_points)
        new_masses = np.random.rand(n_points) * 10
        new_cloud['masses'] = new_masses
        
        plotter.add_mesh(new_cloud, scalars='masses',
                        point_size=15, render_points_as_spheres=True,
                        cmap='plasma', show_scalar_bar=True)
        plotter.add_axes(color='black')
        plotter.add_text("PyVista 3D Concept Space Test (Refreshed)", 
                        position='upper_left', font_size=14)
        
        render_widget.Render()
    
    # Control buttons
    tk.Button(control_frame, text="🔄 Refresh Data", 
             command=refresh_data, font=('Arial', 12)).pack(side=tk.LEFT, padx=5)
    
    tk.Button(control_frame, text="❌ Close", 
             command=root.quit, font=('Arial', 12)).pack(side=tk.RIGHT, padx=5)
    
    # Status label
    status_label = tk.Label(control_frame, 
                           text="✅ PyVista-Tkinter 3D Integration Active - Use mouse to interact!",
                           font=('Arial', 10), fg='green')
    status_label.pack(side=tk.LEFT, padx=20)
    
    # Render initial plot
    plotter.render()
    
    print("🚀 PyVista-Tkinter 3D test window opened!")
    print("   - Use mouse to rotate, zoom, pan")
    print("   - Click 'Refresh Data' to test interactivity")
    print("   - Close window when done testing")
    
    # Start Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    print("🧪 Testing PyVista-Tkinter 3D Integration...")
    create_test_3d_plot()
    print("✅ Test completed!")