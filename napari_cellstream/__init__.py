# napari_cellstream/__init__.py

import torch
from .spectral_analyzer import SpectralWidget
from ._plugin import make_spectral_widget 

__all__ = ["SpectralWidget", "make_spectral_widget"]
