# napari-cellstream

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/status/napari-cellstream)](https://www.napari-hub.org/plugins/napari-cellstream)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-cellstream.svg)](https://pypi.org/project/napari-cellstream)

A napari spectral analyzer for pixel-level cellstream diagnostics. This plugin provides tools for analyzing time-series imagery using Fast Fourier Transforms (FFT) and Continuous Wavelet Transforms (CWT), specifically designed for use with the [cellstream](https://github.com/coylelab/cellstream) image processing toolbox.

---

## Features

### 1. Pixel Inspector
- **Interactive Analysis:** Use `Shift + Click` on any image layer to inspect the temporal behavior of a single pixel.
- **Multi-Domain View:** Instantly visualize:
  - **Time Domain:** Raw intensity over time.
  - **Frequency Domain (FFT):** Power spectrum for identifying dominant frequencies.
  - **Scale-Time Domain (CWT):** Spectrogram showing how frequency content evolves over time.
- **Customizable Wavelets:** Support for various wavelet families (GMW, Morlet, Bump, etc.) with adjustable parameters.

### 2. Spectral Feature Generation
- **FFT Features:** Generate full-image feature maps for amplitude, phase, and Z-scores across specified frequency bins.
- **CWT Features:** Perform block-wise CWT processing to extract features across multiple scales.
- **GPU Acceleration:** Leverage GPU support (CUDA or MPS) for high-performance spectral decomposition.

### 3. Visualization Tools
- **False-Color Spectrum:** Visualize spectral axes (time, frequency, or scale) using color-coded projections.
- **Downsampling:** Efficiently downsample large datasets in time or space for faster processing and visualization.

---

## Installation

You can install `napari-cellstream` via [pip]:

```bash
pip install napari-cellstream
```

To enable GPU acceleration, ensure you have `torch` installed with CUDA support (for NVIDIA GPUs) or that you are on a Mac with Apple Silicon (for MPS support).

---

## Usage

1. **Launch napari:**
   ```bash
   napari
   ```
2. **Open the plugin:**
   Go to `Plugins` -> `napari-cellstream: Spectral Viewer`.
3. **Pixel Inspector:**
   - Click **Activate Pixel Inspector**.
   - Select an image layer with a time dimension (T, C, X, Y or T, X, Y).
   - `Shift + Click` on the image to view the spectral analysis in the sidebar.
4. **Generate Features:**
   - Use the **Generate FFT features** or **Generate CWT features** sections to create new image layers based on spectral analysis.

---

## Dependencies

This plugin is inteneded to be used with [cellstream](https://github.com/coylelab/cellstream) for core image processing utilities.

---

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request or open an issue on GitHub.

## License

Distributed under the terms of the [BSD-3-Clause] license.

[pip]: https://pypi.org/project/pip/
[BSD-3-Clause]: http://opensource.org/licenses/BSD-3-Clause
