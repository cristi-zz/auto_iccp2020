# auto_iccp2020


Supporting code for ICCP 2020 "System Identification using several Deep Learning architectures" paper


## Installation:

Tested on Linux (Ubuntu 16.04+). Possibly runnable under Windows too. It relies on Anaconda environment.

Uses Python and PyTorch and Fast dot AI, v2. 

One must install fast dot ai in DEV mode. Being in dev, things might change so this code will break. 
Create an issue and I'll sort it out eventually.


Install CUDA 10.2: https://developer.nvidia.com/cuda-downloads

Create a Python Env and "compile" Jupyter lab support:

    conda create -y --copy -c conda-forge -c fastai -c pytorch -n f2 python\>=3.7 pytest numpy pandas matplotlib seaborn pillow scikit-learn scipy spacy pywavelets psutil pytest requests pyyaml jupyterlab ipywidgets ipympl nodejs\>=10  pytorch\>=1.3.0 torchvision\>=0.5 fastprogress\>=0.1.22 cudatoolkit 
    conda activate f2
    jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib

Fast dot AI specifics:

    git clone https://github.com/fastai/fastcore.git
    cd fastcore
    pip install -e ".[dev]"


    git clone https://github.com/fastai/fastai2
    cd fastai2
    pip install -e ".[dev]"


