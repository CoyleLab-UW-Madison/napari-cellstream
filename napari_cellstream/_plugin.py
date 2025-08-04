# napari_cellstream/_plugin.py


from napari import current_viewer

def make_spectral_widget(viewer=None):
    print(f"make_spectral_widget called. viewer passed? {viewer is not None}")
    if viewer is None:
        viewer = current_viewer()
        if viewer is None:
            raise RuntimeError("No active viewer found via napari.current_viewer()")
    from .spectral_analyzer import SpectralWidget
    return SpectralWidget(viewer)


# def make_spectral_widget(viewer):
#    from .spectral_analyzer import SpectralWidget
#    return SpectralWidget(viewer)
#
# __all__ = ["make_spectral_widget"]
