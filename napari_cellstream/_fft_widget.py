from magicgui import magicgui
import numpy as np
from napari.layers import Image
from napari import current_viewer
from cellstream.fft import generate_fft_features
import torch


#fft_features_to_process=['full_amplitude', 'normalized_amplitude', 'z_score', 'phase']
@magicgui(
    call_button="Generate FFT features",
   )
def fft_gui_widget(
    normalize_histogram=True,
    max_bin=128,
    batch_size=None,

    return_amplitude: bool = True,
    return_norm_amp: bool = False,
    return_phase: bool = False,
    return_z_score: bool = True
):
    viewer = current_viewer()
    if viewer is None:
        raise RuntimeError("No active napari viewer found")

    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, Image):
        raise RuntimeError("No active image layer selected")

    fft_features_to_process=[]
    if return_amplitude==True:
        fft_features_to_process.append('full_amplitude')
    if return_norm_amp==True:
        fft_features_to_process.append('normalized_amplitude')
    if return_phase==True:
        fft_features_to_process.append('phase')
    if return_z_score==True:
        fft_features_to_process.append('z_score')


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
