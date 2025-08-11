# napari_cellstream/__init__.py

import torch

from ._plugin import make_spectral_widget
from .spectral_analyzer import SpectralWidget

__all__ = ["SpectralWidget", "make_spectral_widget"]
