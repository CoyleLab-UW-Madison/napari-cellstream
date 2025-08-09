from magicgui import magicgui
from napari.types import ImageData
from napari.layers import Image
from napari import current_viewer
from qtpy.QtWidgets import QWidget
import torch
import numpy as np

from cellstream.image import downsample  

@magicgui(
    call_button="Downsample active image",
)
def downsample_gui_widget(
    downsample_by: float = 1,
    is_mask: bool = False,
):
    
    viewer = current_viewer()
    if viewer is None:
        raise RuntimeError("No active napari viewer found")

    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, Image):
        raise RuntimeError("No active image layer selected")

    img = layer.data
    img = torch.from_numpy(img.astype('float32'))
 
    print("Genearting color-coded spectra...")
    ds = downsample(
        tensor=img,
        scale=downsample_by,
        is_mask=is_mask
    )

    ds=ds.detach().numpy()
    return ds