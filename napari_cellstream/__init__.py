# napari_cellstream/__init__.py

__version__ = "0.1.0"

def __getattr__(name):
    if name == "SpectralWidget":
        from .spectral_analyzer import SpectralWidget
        return SpectralWidget
    if name == "make_spectral_widget":
        from ._plugin import make_spectral_widget
        return make_spectral_widget
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = ["SpectralWidget", "make_spectral_widget"]
