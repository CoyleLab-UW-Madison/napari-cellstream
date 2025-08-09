from magicgui import magicgui
from napari.types import ImageData
from napari.layers import Image
from napari import current_viewer
from qtpy.QtWidgets import QWidget
import torch
import numpy as np

from cellstream.image import color_by_axis  

@magicgui(
    call_button="False-color spectrum",
)
def false_color_widget(
    #viewer: "napari.viewer.Viewer",
    #img_layer: Image,
    min_slice: int = 0,
    max_slice: int = 40,
    colormap: str = 'turbo'
):
    
    viewer = current_viewer()
    if viewer is None:
        raise RuntimeError("No active napari viewer found")

    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, Image):
        raise RuntimeError("No active image layer selected")

    img = layer.data
    img = torch.from_numpy(img.astype('float32'))

    #trim array and handle 5D cwt inputs
    img_ndim=img.dim()
    if img_ndim==4:
        img=img[min_slice:max_slice]
    elif img_ndim==5: #coming from generate_cwt_features
        img=img[:,:,min_slice:max_slice,:,:]
 
    print("Genearting color-coded spectra...")
    cc = color_by_axis(
        img=img,
        cmap=colormap,
    )

    cc=cc.detach().numpy()
    return cc