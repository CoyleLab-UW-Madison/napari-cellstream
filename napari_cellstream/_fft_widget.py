from magicgui import magicgui
import numpy as np
from napari.layers import Image
from napari import current_viewer
from cellstream.fft import generate_fft_features
import torch

@magicgui(
    call_button="Generate FFT features",
   )
def fft_gui_widget(
    normalize_histogram=True,
    max_bin=128,
    batch_size=None,
    fft_features_to_process=['z_score']
):
    viewer = current_viewer()
    if viewer is None:
        raise RuntimeError("No active napari viewer found")

    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, Image):
        raise RuntimeError("No active image layer selected")

    image_data = layer.data
    if isinstance(image_data, np.ndarray):
        image_data = torch.from_numpy(image_data.astype('float32'))
    feature_dict = generate_fft_features(
        image_data,
        normalize_histogram=normalize_histogram,
        max_bin=max_bin,
        batch_size=batch_size,
        fft_features_to_process=fft_features_to_process,
    )
    return feature_dict
