# Diffraction and Coherent Propagation
A computational implementation of coherent propagation of complex fields, all written in Python.

## What's inside?
In this package, a set of utilities are provided to simulate coherent signal propagation and diffraction. It is particularly made for free-space optical propagation and holography. However, the tools are compatible with 1D and 2D data structures and can be potentially used for any sort of spatially-coherent signals.

It is intended to be modular and simple to understand. The codes are GPU-friendly, and they support batch processing. Angular spectrum algorithm is the primary work horse for signal propagtion though any custom solver could be given. Aside Numpy and TensorFlow backends which are provided with the package, any computational means could be implemented to process tensor operations.

It also includes:
- a simple yet useful data pipeline which supports images only.
- Gerchberg-Saxton multi-distance phase recovery algorithm. It can be easily tweaked to support other variations of signal e.g. wavelength.

## Installation
To install the package, run:
```
python -m pip install fringe
<<<<<<< HEAD
=======
```
The example files are not included in the package. To import them, clone the repository. In the git bash, run:
```
$ git clone https://github.com/farhadnkm/Fringe.Py
>>>>>>> 5bf76ba15b5f35a6c28f9681578d8f1f17659458
```
The example files are not included in the package and should be downloaded separately. Also they require *matplotlib* to show plots.

## How to Use
1. Import or create data

For images:
```
import numpy as np
from fringe.utils.io import import_image
from fringe.utils.modifiers import ImageToArray, Normalize, MakeComplex
```
Images need to be standardized, normalized, and casted to complex data type. *Modifiers* are tools made for this purpose which apply these operations on import.
```
p1 = ImageToArray(bit_depth=16, channel='gray', crop_window=None, dtype='float32')
p2 = Normalize(background=np.ones((512, 512)))
p3 = MakeComplex(set_as='amplitude', phase=0)

obj = import_image("images/squares.png", preprocessor=[p1, p2, p3])
```
2. Propagate

*Solvers* contain propagation algorithms and can be called by *solver.solve*. In particular, angular Spectrum algorithm convolves the input field with a free-space propagtor function which depends on *wavelength λ* (or *wavenumber k=2π/λ*) and distance *z*.
```
from fringe.solvers.AngularSpectrum import AngularSpectrumSolver as AsSolver

solver = AsSolver(shape=obj.shape, dr=1, is_batched=False, padding="same", pad_fill_value=0, backend="Numpy")
rec = solver.solve(hologram, k=2*np.pi/500e-3, z=-300)
amp, phase = np.abs(rec), np.angle(rec)
```



Example notebooks provide further details with 1D and 2D signal propagtion, GPU acceleration and batch processing, and phase recovery.

## Outcomes

A Hologram:

<img src="images/hologram_preview.png" width="300">

Reconstructed Amplitude and phase images obtained by back propagation:

<img src="images/exports/bp_amplitude.png" width="300"> <img src="images/exports/bp_phase.png" width="300">

Reconstructed Amplitude and phase images obtained by MHPR method using 8 axially displaced holograms:

<img src="images/exports/mhpr_amplitude.png" width="300"> <img src="images/exports/mhpr_phase.png" width="300">
