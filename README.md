# datamovies: Introduction
This package includes a couple of demo codes for sonifying data, written by Ben Holtzman (https://www.ldeo.columbia.edu/~benh/) and others.

The codes in this package are in the form of Jupyter Notebooks, which are becoming the most popular way to write and share Python code within the scientific community. The easiest way to install Jupyter Notebook is by installing Anaconda – it comes included with Anaconda. To download Anaconda, visit https://www.anaconda.com/download/.

Once downloaded, run the Jupyter Notebook by typing in the Mac Terminal (or Command Prompt for Windows):

`jupyter notebook`

## Required Python packages for this demo
- [**numpy**        ](https://anaconda.org/anaconda/numpy)
- [**scipy**        ](https://anaconda.org/anaconda/scipy)
- [**matplotlib**   ](https://anaconda.org/conda-forge/matplotlib)
- [**librosa**      ](https://anaconda.org/conda-forge/librosa)
- [**resampy**      ](https://anaconda.org/conda-forge/resampy)

The easiest way to download these packages is using Anaconda (see above). Follow the link for each package name above and copy-paste the corresponding command into a terminal window. For example, to install **numpy** type this into the Mac Terminal:

`conda install -c anaconda numpy`

To make sure the packages were successfully installed, type:

`conda list`

## Clone repository
To access the codes from your machine, open the Mac Terminal and type:

`git clone https://github.com/jbrussell/datamovies`

Alternatively, click *"Clone or download"* at the top right corner of this page and *"Download ZIP"*.

## [1_notebooks](https://github.com/jbrussell/datamovies/tree/master/1_notebooks): where the fun happens!
This directory contains the demo "notebooks" for sonifying data.

- **NB00_make_simple_sounds.ipynb** : Demonstrates sonification of time-varying (sinusoidal) signals
- **NB01_DirectSonification.ipynb** : Reads in a real 14 hour long seismogram (ground motion) from the 2011 Tohoku earthquake and turns it into sound! The .wav soundfile is output to the **3_output** directory and can be played using most audio software such as iTunes or Audacity.

*Note: added .py versions of the notebooks in case Jupyter Notebook is not installed*