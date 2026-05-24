# napari_cellstream/_plugin.py

from qtpy.QtWidgets import QMessageBox
from napari import current_viewer
import os


import logging

logger = logging.getLogger(__name__)

_USE_GPU = None

def make_spectral_widget(viewer=None):
    global _USE_GPU
    
    #handle viewer injection
    logger.info("make_spectral_widget called. viewer passed? %s", viewer is not None)
    if viewer is None:
        viewer = current_viewer()
        if viewer is None:
            raise RuntimeError("No active viewer found via napari.current_viewer()")


    #set ssq_gpu environment if not already set
    if _USE_GPU is None:
        msg = QMessageBox()
        msg.setWindowTitle("Enable GPU Support?")
        msg.setText("Would you like to enable GPU acceleration (if available)?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)
        ret = msg.exec_()

        if ret == QMessageBox.Yes:
            _USE_GPU = True
        else:
            _USE_GPU = False

    if _USE_GPU:
        os.environ["SSQ_GPU"] = "1"
    else:
        os.environ["SSQ_GPU"] = "0"

    # Apply torch tensor patch for napari
    try:
        from cellstream.viz import patch_napari_for_torch
        patch_napari_for_torch()
        logger.info("Applied cellstream.viz.patch_napari_for_torch()")
    except ImportError:
        logger.warning("Could not import cellstream.viz.patch_napari_for_torch()")
    except Exception as e:
        logger.error(f"Failed to apply torch patch: {e}")

    # Importing widget until *after* setting the ssq_gpu env var

    from .spectral_analyzer import SpectralWidget
    return SpectralWidget(viewer,use_gpu=_USE_GPU)
