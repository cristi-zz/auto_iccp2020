Support repository for:
# Inductive biases and Self Supervised Learning in modelling a physical heating system

 - Paper: https://arxiv.org/abs/2104.11478
 - Video: https://odysee.com/@ml-visoft:d/00_Delay_nets:e?r=Azj9J3xvCfqpXHyNdrubia8ymKELp28j
 - Data:  https://github.com/cristi-zz/auto_iccp2020/tree/master/data_sample

## Overview and motivation

Suppose you want to control a system in a way that a certain measurement (eg temperature) stays at a fixed value. If your system 
reacts to its inputs with a considerable delay (eg heating element has high inertia) classical alternatives will induce 
oscillations in the system. One very interesting alternative is to use a Model Predictive Control loop. This MPC, takes
a model (how the inputs are influencing the output) and gives it a lot of potential commands. The model will output some
predictions about the potential outputs, and the MPC can decide which is the best course of action with respect to some 
constraints and objectives. 

Present paper focuses only on the modelling part of the MPC loop and, of course, tries to model the system using the current
hype (as of 2021), neural networks (NN). A NN that can be used in a MPC loop must be able to take some inputs (info about
the current state and a list of future commands) and predict some outputs (future values of some targets). If the network
has to ingest only the current state and output one (or more) future states, we would be in classical supervised learning
framework. However, adding the future commands that need to be evaluated complicates things. 

Previous work took some popular NN architectures and "tuned" them, so they can ingest and process data in above format. 
One architecture took the "SOTA crown", that is, a Recurrent Neural Network with an Attention mechanism between the encoding
and decoding layers. 

Current paper designs a network architecture guided by how the actual thermal system works. There are delays, unknown states,
unknown variable interactions and it approximates them using several inductive biases. The delays are modelled as a convolution and
the interactions are discovered using a fully connected (sub)network. Moreover, there is an assumption that the possible delays
are system dependent (false, for a thermal system, hence we introduce bias in the model) and once these delays are removed,
the feature interaction can be learned from the data. The motivation for these biases is that Recurrent networks are serial
networks, very hard to parallelize. Each time step depends on the previous time step, and it is not possible to compute
all time steps in parallel. Introduced biases try to separate the time dependent component from feature dependent 
component to achieve a faster-to-train network by leveraging massive parallel computation cores available.

## Data

The training data is from a household. 15 sensors were placed on various components of the heating system (and not only).
Unfortunately there are several privacy issues that prevent us from putting everything online. However, a small, curated
and slightly processed dataset is made public.

There are many features removed (eg is tap water faucet open?) and only few days exported in the public sets. 
The train/validation samples presented here are from train/validation period used in the paper. To compensate for reduced
data diversity I augmented the data by a denser oversampling. 

The models presented in the paper are too big and overfit on such small dataset. I hand tuned few networks, basically by
heavily trim the internal FCN component until there was some learning.

A reader that wants to test some other models on the data should also put some computing power in a hyperparemeter 
search for the above models.

Head to the ``data_sample`` folder in this repo and to ``src`` for examples on how to load and use it.

## Installation:

Tested on Linux (Ubuntu 18.04+). Possibly runnable under Windows too. It relies on Anaconda environment.

Uses Python and PyTorch and fast.ai . 

Install CUDA 10.2: https://developer.nvidia.com/cuda-downloads

Create a Python Env and "compile" Jupyter lab support:

    conda create -y --copy -c fastai -c pytorch -n f2i python=3.8 pytest numpy=1.19 pandas=1.2 matplotlib=3.3 seaborn pillow scikit-learn scipy spacy pywavelets psutil pytest requests pyyaml jupyterlab ipywidgets ipympl nodejs\>=10  pytorch=1.8.1 fastprogress\>=0.1.22 fastai cudatoolkit\>=10.2 
    conda activate f2i
    jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib


Now clone this repo and start the ``jupyter lab`` in ``src`` folder.

Head to the  ``src`` folder for a short explanation about the experiments. The notebooks are exported as PDFs for quick inspection
without the need to install and run the whole environment.

### Note: 

For the previous work, "System Identification using several Deep Learning architectures" submitted at ICCP 2020 please see: https://github.com/cristi-zz/auto_iccp2020/tree/iccp_2020  
