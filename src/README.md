## Supporting code for "Inductive biases and Self Supervised Learning in modelling a physical heating system"

Here, one can find:

 * The code used to create the models
 * Experimental setup (learning rates, callbacks, etc) to train the networks.
 * A runnable example that produces results on the available public dataset. 

### network_definitions.py

Code with the models and data preprocessing. Check the documentation inside for how the data is formatted and how
the models are generated.

### demo_training_small.ipynb and demo_training_small.pdf

A runnable example with some small models. They are trained and evaluated on the published dataset. The goal of the 
training was to "beat" a trivial classifier, so they are not meant to be used as baselines for future developments. 

### demo_training_paper.ipynb and demo_training_paper.pdf

The notebook shows how the experiments were instantiated in the paper: learning rates, callbacks, LR schedulers, 
early stopping parameters, optimizers, data preprocessing etc. All these little nifty details can impact the learning
dynamics and performance.


