from magicgui import magicgui
from napari.types import ImageData
from napari.layers import Image
from qtpy.QtWidgets import QWidget
import torch
import numpy as np

from cellstream.cwt.utils import generate_cwt_image_cellstreams

@magicgui(
    call_button="Generate CWT Features",
    blocks={"min": 1, "max": 100000},
)
def generate_cwt_features_widget(
    #viewer: "napari.viewer.Viewer",
    img_layer: Image,
    min_scale: int = 80,
    max_scale: int = 180,
    num_filter_banks: int = 1,
    carrier_channel: int = 0,
    blocks: int = 50,
    normalize_amplitudes: bool = False,
    use_gpu: bool = False,
    bank_method: str = 'max_pool',
    downsample_by: float = 1.0,
    normalize_histogram: bool = True,
    mean_center: bool = False,
    return_amplitude: bool = True,
    return_scales: bool = True,
    return_phase: bool = False,
    return_z_score: bool = True
):
    img = img_layer.data
    if img.ndim != 4:
        print("Expected image shape (T, C, X, Y)")
        return
    T, C, X, Y=img.shape

    # Convert numpy to torch tensor
    img_tensor = torch.from_numpy(img.astype('float32'))

    # Build channel_outputs dynamically — for now just do amp, freq, phase for channel 0
    # Could make this user-configurable later
    channel_outputs=dict()
    for c in range(C):
        channel_returns=list()
        if return_amplitude==True:
            channel_returns.append('amp')
        if return_scales==True:
            channel_returns.append('freq')
        if return_phase==True:
            channel_returns.append('phase')
        if return_z_score==True:
            channel_returns.append('z_score')
        channel_outputs[c]=channel_returns
    #channel_outputs = {0: ['amp', 'freq', 'phase']}

    print("Running CWT blockwise feature generation...")
    results = generate_cwt_image_cellstreams(
        img=img_tensor,
        min_scale=min_scale,
        max_scale=max_scale,
        num_filter_banks=num_filter_banks,
        normalize_amplitudes=normalize_amplitudes,
        blocks=blocks,
        use_gpu=use_gpu,
        bank_method=bank_method,
        downsample_by=downsample_by,
        normalize_histogram=normalize_histogram,
        mean_center=mean_center,
        carrier_channel=carrier_channel,
        channel_outputs=channel_outputs,
    )

    return results