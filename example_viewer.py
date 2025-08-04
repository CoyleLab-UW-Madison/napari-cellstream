# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 14:49:16 2025

@author: smcoyle
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 23:43:23 2025

@author: smcoyle
"""
import torch
import napari
import cellstream
#from napari_cellstream import SpectralWidget

# Main workflow
if __name__ == "__main__":
    # Load and preprocess image
    img = cellstream.image.load_image("example_timeseries_mini_0.tif")
    img = cellstream.image.normalize_histogram(img)
    img = img.detach().numpy()
    
    # Create viewer and add image
    viewer = napari.Viewer()
    viewer.add_image(img)
    
    # Create the inspector widget
    # widget = SpectralWidget(viewer)
    # viewer.window.add_dock_widget(widget, name="Pixel Inspector")
    
    # # Start the application
    # napari.run()