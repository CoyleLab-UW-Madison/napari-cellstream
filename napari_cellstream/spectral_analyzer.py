# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 15:35:18 2025

@author: smcoyle
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import nd2
import tifffile
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                           QComboBox, QSpinBox, QCheckBox, QHBoxLayout,
                           QGroupBox, QDoubleSpinBox, QFormLayout,QFileDialog)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from ssqueezepy import cwt
from ._fft_widget import fft_gui_widget
from ._cwt_widget import generate_cwt_features_widget

# Define wavelet parameters with default values and ranges
WAVELET_PARAMS = {
    'gmw': { "gamma": (3,0,20,1),
             "beta": (3 ,0,20,1)
            },
    'morlet': { "mu": (5,0,20,1)
            },
    
    
    'bump': { "mu": (5,0,20,1),
             "s": (3 ,0,20,1),
             "om": (5,0,20,1)
            },
    
    'cmhat': { "mu": (5,0,20,1),
             "s": (3 ,0,20,1)
             },
    
    'hhhat': { "mu": (5,0,20,1)
            }}

class SpectralWidget(QWidget):
    def __init__(self, napari_viewer,use_gpu=False):
        super().__init__()
        self.viewer = napari_viewer
        self.use_gpu=use_gpu
        self.cid = None
        self.canvas = None
        self.last_x = None
        self.last_y = None
        self.last_layer = None
        
        #timelines
        self.current_time_index = 0
        self.time_cursor_lines = []
        self.viewer.dims.events.current_step.connect(self.update_time_cursor) #time line
        
        # Default settings
        self.wavelet = "morlet"
        self.nv = 32
        self.do_plot_zscore = True
        self.wavelet_params = {}
        
        # Create controls
        self.create_controls()
        
        # Activation toggle
        self.activate_button = QPushButton("Activate Pixel Inspector")
        self.activate_button.setCheckable(True)
        self.activate_button.setChecked(True)
        self.activate_button.toggled.connect(self.toggle_activation)
        
        # Status indicator
        self.status_label = QLabel("Status: ACTIVE - Shift+Click on image")
        
        # Plot container
        self.plot_container = QWidget()
        self.plot_container.setLayout(QVBoxLayout())
        self.plot_container.setMinimumSize(100, 100)
        
        #fft widget 
        self.fft_gui = fft_gui_widget
        self.fft_gui.called.connect(self.handle_fft_result)
        self.plot_container.layout().addWidget(self.fft_gui.native)
        
        fft_group = QGroupBox("     Generate FFT features")
        fft_layout = QVBoxLayout()
        fft_layout.addWidget(self.fft_gui.native)
        fft_group.setLayout(fft_layout)

        #cwt widget 
        self.cwt_gui = generate_cwt_features_widget
        self.cwt_gui.use_gpu.value=self.use_gpu
        #self.cwt_gui.wavelet_tuple=self.get_wavelet_tuple() ##fix this later
        self.cwt_gui.called.connect(self.handle_cwt_result)
        self.plot_container.layout().addWidget(self.cwt_gui.native)
        
        cwt_group = QGroupBox("     Generate CWT features")
        cwt_layout = QVBoxLayout()
        cwt_layout.addWidget(self.cwt_gui.native)
        cwt_group.setLayout(cwt_layout)
        
        # Load image button
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.open_file_dialog)

        # Save plots button
        self.save_plot_button = QPushButton("Save Plot")
        self.save_plot_button.clicked.connect(self.save_figure)
        self.plot_container.layout().addWidget(self.save_plot_button)
        
        # Left column: Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        left_layout.addWidget(self.load_button)
        left_layout.addWidget(self.controls_group)
        left_layout.addWidget(fft_group)
        left_layout.addWidget(cwt_group)
        
        # Right column: Plots
        right_panel = self.plot_container  # already a QWidget with a VBoxLayout
        
        ### Main layout: horizontal split ###
        main_layout = QHBoxLayout()
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        self.setLayout(main_layout)
        
        self.toggle_activation(True)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.tif *.tiff *.png *.jpg *.jpeg *.nd2)")
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, path: str):
        print(f"Loading image from: {path}")
        iname, iext = path.split('.')
        if iext == 'nd2':
            image = nd2.imread(path)
        elif iext == 'tif':
            image = tifffile.imread(path)
        else:
            print("Invalid type")
            return
        #check dimension -- insert dummy channel dimension if need by
        if image.ndim==3:
            image=np.expand_dims(image,axis=1)
            print("Inserting channel dimension...")
        
        self.viewer.add_image(image, name=iname)
        return
    
    def save_figure(self):
        if self.canvas is None:
            print("No plot to save.")
            return
    
        file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Figure", "", 
                "spectral_plot.svg",  # Default filename with .svg
                "SVG Files (*.svg);;PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)"
            )
    
        if file_path:
            self.canvas.figure.savefig(file_path, bbox_inches='tight', dpi=300)
            print(f"Figure saved to: {file_path}")
        
    def handle_fft_result(self, result):
        if not isinstance(result, dict):
            print("FFT did not return a valid result")
            return

        for key, data in result.items():
            name = f"FFT_{key}"
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            self.viewer.add_image(data, name=name)

    def handle_cwt_result(self,results):
        # Add results to Napari viewer
        for chan, feature_dict in results.items():
            for feat_name, feat_tensor in feature_dict.items():
                layer_name = f"cwt_c{chan}_{feat_name}"
                self.viewer.add_image(
                    feat_tensor.numpy(),  # already (T, num_filter_banks, X, Y)
                    name=layer_name,
                    scale=[20,1,1],
                    metadata={"source": "cwt", "channel": chan, "feature": feat_name}
            )

    def create_controls(self):
        """Create the parameter controls"""
        self.controls_group = QGroupBox("     CWT Parameters")
        controls_layout = QVBoxLayout()
        
        # Wavelet selector
        wavelet_layout = QHBoxLayout()
        wavelet_layout.addWidget(QLabel("Wavelet:"))
        self.wavelet_combo = QComboBox()
        self.wavelet_combo.addItems(list(WAVELET_PARAMS.keys()))
        self.wavelet_combo.setCurrentText(self.wavelet)
        self.wavelet_combo.currentTextChanged.connect(self.wavelet_changed)
        wavelet_layout.addWidget(self.wavelet_combo)
        controls_layout.addLayout(wavelet_layout)
        
        # Wavelet parameters container
        self.params_container = QGroupBox("Wavelet Parameters")
        self.params_layout = QFormLayout()
        self.params_container.setLayout(self.params_layout)
        controls_layout.addWidget(self.params_container)
        self.create_wavelet_params_controls()
        
        # Scales selector
        nv_layout = QHBoxLayout()
        nv_layout.addWidget(QLabel("Number of Scales:"))
        self.nv_spin = QSpinBox()
        self.nv_spin.setRange(1, 64)
        self.nv_spin.setValue(self.nv)
        self.nv_spin.valueChanged.connect(self.nv_changed)
        nv_layout.addWidget(self.nv_spin)
        controls_layout.addLayout(nv_layout)
        
        # Zscore scale checkbox
        self.zscore_check = QCheckBox("Normalize Spectra")
        self.zscore_check.setChecked(self.do_plot_zscore)
        self.zscore_check.stateChanged.connect(self.zscore_changed)
        controls_layout.addWidget(self.zscore_check)
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh Plots")
        self.refresh_button.clicked.connect(self.refresh_plots)
        controls_layout.addWidget(self.refresh_button)
        
        self.controls_group.setLayout(controls_layout)
    
    def create_wavelet_params_controls(self):
        """Create controls for the current wavelet's parameters"""
        # Clear existing controls
        while self.params_layout.rowCount() > 0:
            self.params_layout.removeRow(0)
        
        # Get parameters for current wavelet
        params = WAVELET_PARAMS.get(self.wavelet, {})
        self.wavelet_params = {}
        
        # Create controls for each parameter
        for param_name, param_config in params.items():
            # Unpack configuration: (default, min, max, step)
            default, min_val, max_val, step = param_config
            
            # Create appropriate spin box
            if isinstance(default, int):
                spinbox = QSpinBox()
                spinbox.setRange(int(min_val), int(max_val))
                spinbox.setSingleStep(int(step))
            else:
                spinbox = QDoubleSpinBox()
                spinbox.setRange(min_val, max_val)
                spinbox.setSingleStep(step)
                spinbox.setDecimals(2)
            
            spinbox.setValue(default)
            spinbox.valueChanged.connect(self.param_changed)
            
            # Store the control and its parameter name
            self.wavelet_params[param_name] = spinbox
            self.params_layout.addRow(QLabel(f"{param_name}:"), spinbox)
        
        # Show/hide container based on whether there are parameters
        self.params_container.setVisible(len(params) > 0)
    
    def wavelet_changed(self, value):
        self.wavelet = value
        self.create_wavelet_params_controls()
        self.refresh_plots()
    
    def param_changed(self, value):
        self.refresh_plots()
    
    def nv_changed(self, value):
        self.nv = value
        self.refresh_plots()
    
    def zscore_changed(self, state):
        self.do_plot_zscore = (state == 2)  # 2 is checked
        self.refresh_plots()
    
    def refresh_plots(self):
        """Refresh plots with current settings"""
        if self.last_layer and self.last_x is not None and self.last_y is not None:
            self.process_pixel(self.last_layer, self.last_x, self.last_y)
    
    def get_wavelet_tuple(self):
        """Create the wavelet tuple with current parameters"""
        params_dict = {}
        for param_name, spinbox in self.wavelet_params.items():
            params_dict[param_name] = spinbox.value()
        return (self.wavelet, params_dict)
    
    def toggle_activation(self, active):
        if active:
            if self.cid is None:
                self.cid = self.viewer.mouse_drag_callbacks.append(self.on_click)
            self.status_label.setText("Status: ACTIVE - Shift+Click on image")
            self.activate_button.setText("Deactivate Pixel Inspector")
        else:
            if self.cid is not None:
                self.viewer.mouse_drag_callbacks.remove(self.cid)
                self.cid = None
            self.status_label.setText("Status: INACTIVE - Click button to activate")
            self.activate_button.setText("Activate Pixel Inspector")

    def update_time_cursor(self, event=None):
        if self.canvas is None or not hasattr(self, "time_cursor_lines"):
            return
    
        current_t = self.viewer.dims.current_step[0]  # assumes time axis is 0
        self.current_time_index = current_t
    
        for line in self.time_cursor_lines:
            line.set_xdata([current_t])
    
        self.canvas.draw_idle()
    
    def on_click(self, viewer, event):
        """Handle click events with Shift modifier"""
        if event.button != 1 or "Shift" not in event.modifiers:
            return
            
        try:
            layer = viewer.layers.selection.active
            if layer is None:
                return
                
            # Convert click position to data coordinates
            pos_data = layer.world_to_data(event.position)
            spatial_coords = pos_data[-2:]
            x, y = map(int, spatial_coords)
            
            # Store the clicked position for refreshing
            self.last_layer = layer
            self.last_x = x
            self.last_y = y
            
            # Process the pixel data
            self.process_pixel(layer, x, y)
                
        except Exception as e:
            print(f"Error processing click: {e}")

    def process_pixel(self, layer, x, y):
        """Process and display data for a single pixel"""
        # Get and reshape data
        data = layer.data
        if data.ndim == 3:
            data = data[:, np.newaxis, ...]  # Add channel dimension
        T, C, X, Y = data.shape
        
        # Validate coordinates
        if not (0 <= x < X and 0 <= y < Y):
            print(f"Coordinates out of bounds: ({x}, {y})")
            return
            
        # Clear previous plot
        if self.canvas:
            self.plot_container.layout().removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None
       
       # print("Initializing plots...")
        plt.style.use('dark_background')
        # Create new figure
        fig = Figure(figsize=(8, 8))
        fig.set_layout_engine(None)
        self.canvas = FigureCanvas(fig)
        self.plot_container.layout().addWidget(self.canvas)
        
        # Create subplots: 3 rows (time, FFT, CWT) x C columns
        if C > 1:
            axes = fig.subplots(3, C, squeeze=False)
        else:
            axes = fig.subplots(3, 1, squeeze=False)
        
        for c in range(C):
            # Extract and normalize time series
            ts = data[:, c, x, y].astype(np.float64)
            ts -= np.mean(ts)
            
            # Time domain plot
            #print("Plotting time domain")
            ax = axes[0, c] if C > 1 else axes[0, 0]
            ax.clear()
            ax.plot(ts)
            ax.set_title(f"Channel {c} - Time Domain" if C > 1 else "Time Domain")
            ax.set_xlabel("Time")
            
            cursor = ax.axvline(self.current_time_index, color='red', linestyle='dotted')
            self.time_cursor_lines.append(cursor)

            #print("Plotting freq domain")
            # Frequency domain plot
            ax = axes[1, c] if C > 1 else axes[1, 0]
            ax.clear()
            fft = np.abs(np.fft.rfft(ts))
            if self.do_plot_zscore:
                fft_mean=np.mean(fft)
                fft_std=np.std(fft)
                fft = (fft-fft_mean)/fft_std
                
            ax.plot(fft,color='#FF91A4')
            ax.set_title("Frequency Domain")
            ax.set_xlabel("FFT bin number")
            
            # CWT plot
            ax = axes[2, c] if C > 1 else axes[2, 0]
            ax.clear()
            
            # Get wavelet tuple with current parameters
            wavelet_tuple = self.get_wavelet_tuple()
            
            #print("Plotting cwt domain")
            # Compute CWT with the selected wavelet and parameters
            Wx, _ = cwt(ts, wavelet=wavelet_tuple, nv=self.nv)

            if isinstance(Wx, torch.Tensor):
                cwt_mag = Wx.abs().cpu().numpy()
            elif isinstance(Wx, np.ndarray):
                cwt_mag = np.abs(Wx)

            if self.do_plot_zscore:
                cwt_mag_mean=np.mean(cwt_mag,axis=0)
                cwt_mag_std=np.std(cwt_mag,axis=0)
                cwt_mag=(cwt_mag-cwt_mag_mean)/cwt_mag_std
           
           # print("Finalizing plots...")
            im = ax.imshow(cwt_mag, aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f"CWT: {wavelet_tuple[0]}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Scale")
            
            cursor = ax.axvline(self.current_time_index, color='red', linestyle='dotted')
            self.time_cursor_lines.append(cursor)
            
            #fig.colorbar(im, ax=ax)
            
        fig.tight_layout()
        self.canvas.draw()

    def closeEvent(self, event):
        """Clean up when widget is closed"""
        if self.cid is not None:
            self.viewer.mouse_drag_callbacks.remove(self.cid)
        super().closeEvent(event)

