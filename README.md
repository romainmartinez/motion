<img
  src="https://raw.githubusercontent.com/pyomeca/design/master/logo/logo_plain_doc.svg?sanitize=true"
  alt="logo"
/>

<p align="center">
  <a href="https://github.com/romainmartinez/motion/actions"
    ><img
      alt="Actions Status"
      src="https://github.com/romainmartinez/motion/workflows/CI/badge.svg"
  /></a>
  <a href="https://coveralls.io/github/romainmartinez/motion?branch=master"
    ><img
      alt="Coverage Status"
      src="https://coveralls.io/repos/github/romainmartinez/motion/badge.svg?branch=master"
  /></a>
  <a href="https://anaconda.org/conda-forge/pyomeca"
    ><img
      alt="License: MIT"
      src="https://anaconda.org/conda-forge/pyomeca/badges/license.svg"
  /></a>
  <a href="https://anaconda.org/conda-forge/pyomeca"
    ><img
      alt="PyPI"
      src="https://anaconda.org/conda-forge/pyomeca/badges/latest_release_date.svg"
  /></a>
  <a href="https://anaconda.org/conda-forge/pyomeca"
    ><img
      alt="Downloads"
      src="https://anaconda.org/conda-forge/pyomeca/badges/downloads.svg"
  /></a>
  <a href="https://github.com/psf/black"
    ><img
      alt="Code style: black"
      src="https://img.shields.io/badge/code%20style-black-000000.svg"
  /></a>
</p>

Pyomeca is a python library allowing you to carry out a complete biomechanical analysis; in a simple, logical and concise way.

## Pyomeca documentation

See Pyomeca's [documentation site](https://romainmartinez.github.io/motion).

## Example

Here is an example of a complete EMG pipeline in just one command:

```python
from motion import Analogs
```

## Features

- Object-oriented architecture where each class is associated with common and specialized functionalities:
  - **Markers3d**: 3d markers positions
  - **Analogs3d**: analogs (emg, force or any analog meca)
  - **GeneralizedCoordinate**: generalized coordinate (joint angle)
  - **RotoTrans**: roto-translation matrix


- Specialized functionalities including processing routine commonly used in biomechanics: filters, normalization, onset detection, outliers detection, derivative, etc.


- Each functionality can be chained. In addition to making it easier to write and read code, it allows you to add and remove analysis steps easily (such as Lego blocks).


- Each class inherits from a numpy array, so you can create your own analysis step easily.


- Easy reading and writing interface to common files in biomechanics (`.c3d`, `.csv`, `.xlsx`, `.mat`, `.sto`, `.trc`, `.mot`):


- Common linear algebra routine implemented: get Euler angles to/from roto-translation matrix, create a system of axes, set a rotation or translation, transpose or inverse, etc.

## Installation

### Using Conda

First, install [miniconda](https://conda.io/miniconda.html) or [anaconda](https://www.anaconda.com/download/).
Then type:

```bash
conda install pyomeca -c conda-forge
```

### Using pip

First, you need to install python, swig and numpy. 
Then, follow the instructions to compile [ezc3d](https://github.com/pyomeca/ezc3d).
Finally, install pyomeca with:

```bash
pip install git+https://github.com/pyomeca/pyomeca/`
```

## Integration with other modules

Pyomeca is designed to work well with other libraries that we have developed:

- [pyosim](https://github.com/pyomeca/pyosim): interface between [OpenSim](http://opensim.stanford.edu/) and pyomeca to perform batch musculoskeletal analyses
- [ezc3d](https://github.com/pyomeca/ezc3d): Easy to use C3D reader/writer in C++, Python and Matlab
- [biorbd](https://github.com/pyomeca/biorbd): C++ interface and add-ons to the Rigid Body Dynamics Library, with Python and Matlab binders.

## Bug Reports & Questions

Pyomeca is Apache-licensed and the source code is available on [GitHub](https://github.com/pyomeca/pyomeca). If any questions or issues come up as you use pyomeca, please get in touch via [GitHub issues](https://github.com/pyomeca/pyomeca/issues). We welcome any input, feedback, bug reports, and contributions.
