# napari_cellstream/_plugin.py

import os

import torch.cuda
from napari import current_viewer
from qtpy.QtWidgets import QMessageBox


def make_spectral_widget(viewer=None):
    # handle viewer injection
    print(f"make_spectral_widget called. viewer passed? {viewer is not None}")
    if viewer is None:
        viewer = current_viewer()
        if viewer is None:
            raise RuntimeError("No active viewer found via napari.current_viewer()")

    # set ssq_gpu environment
    msg = QMessageBox()
    msg.setWindowTitle("Enable GPU Support?")
    msg.setText("Would you like to enable GPU acceleration (if available)?")
    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    msg.setDefaultButton(QMessageBox.Yes)
    ret = msg.exec_()

    if ret == QMessageBox.Yes and torch.cuda.is_available():  # SSQ_GPU needs cuda
        os.environ["SSQ_GPU"] = "1"
        use_gpu = True
    elif ret == QMessageBox.Yes:
        use_gpu = True
        print("Cuda support not detected. Some features will be unavailable.")
    else:
        os.environ["SSQ_GPU"] = "0"
        use_gpu = False

    # Importing widget until *after* setting the ssq_gpu env var

    from .spectral_analyzer import SpectralWidget

    return SpectralWidget(viewer, use_gpu=use_gpu)
