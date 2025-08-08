from magicgui import magicgui
import numpy as np
from napari.layers import Image
from napari import current_viewer

from cellstream.fft import generate_fft_features
from cellstream.image import downsample
import torch


#fft_features_to_process=['full_amplitude', 'normalized_amplitude', 'z_score', 'phase']
@magicgui(
    call_button="Generate FFT features",
    blocks={"min": 1, "max": 100000}
   )
def fft_gui_widget(
    normalize_histogram=True,
    max_bin=128,
    use_gpu: bool = False,
    blocks: int = 1,
    downsample_by: float=1,

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
        
        
    if use_gpu==True:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # For Macs with M1/M2 GPUs
        else:
            device = torch.device("cpu")
            print("No GPU detected; switching to CPU mode...")
    else:
        device = torch.device("cpu")
        
        

    print(f"Performing blocked FFT processing using device: {device}")
        
    image_data = layer.data
    
    num_pixels=image_data.size
    batch_size=int(num_pixels/blocks)
    
    if isinstance(image_data, np.ndarray):
        image_data = torch.from_numpy(image_data.astype('float32'))
    
    if downsample_by < 1:
        print(f"Downsampling image by {downsample_by}...")
        image_data=downsample(image_data,downsample_by)
        
    feature_dict = generate_fft_features(
        image_data,
        normalize_histogram=normalize_histogram,
        max_bin=max_bin,
        batch_size=batch_size,
        fft_features_to_process=fft_features_to_process,
        device=device
    )
    
    return feature_dict
