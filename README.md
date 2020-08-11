# System Identification using several Deep Learning architectures


Supporting code for "Evaluation of Deep Learning architectures for System Identification. Applications for a household 
heating system" paper submitted at ICCP 2020. [Download draft paper.](http://users.utcluj.ro/~visoft/publications.php?action=getpdf&artid=iccp_20)

Suppose you want to control a system such that, a certain measurement (eg temperature) stays at a fixed value. Usually one can
employ a PID controller. This, will react to the difference between the desired value and current measurement, to the speed of
change and to the small systematic errors that accumulate. Pretty handy tool in automation.

If your system have a delay (eg heating element have high inertia) and use a PID, you will get heat into the system when the temperature is quite low.
And the heating element will continue to heat (due to inertia) well after the temperature rose to the desired value.

There is another type of control called Model Predictive Control. Here, different commands are applied to a "model" of the
system and the *best* command set (wrt to some criteria) is sent to the actual system. **In theory**, such a controller
could start the heating before the temperature gets under the desired value. So, by the time the temperature at the sensor
is lower than the threshold the heating element is already radiating and it radiates just enough, without overheating.


Well, that's the theory. In automation field, this "model" is vital for a proprer MPC operation. This is where present research
comes into play.

Now, we have tons of architectures and models in Machine Learning. Are they any good at modelling a real system? A model that 
could be used inside a Smith controller?

This paper is a first step towards this. A regular time series multivariate model takes previous sensor values and 
previous commands sent to the system and predicts one or more future values of a target.

What is special about a model used in an MPC is that it also need to consider some future commands that will be sent
to the system. So it have to predict some future values based on the previous state AND some new commands that will be sent.

The NN architectures on the other hand have the advantage that some non command inputs, that depend on the environment
can also be predicted, especially if their behavior follow a pattern (eg day/night temperature cycles). This prediction
happens intrinsically to the model and is used to make better predictions given a certain set of future commands. 

Long story short, here we adapted several well known architectures to be able to ingest such data AND be able to learn
something from it.

There are six architectures and three datasets in the paper. Unfortunately the most interesting dataset has been collected from a 
private household and contain too much personal information. It can't be shared.

The networks developed here can be used in other fields, where one needs to take some decisions given present
data. An example is with data compression. Suppose that one needs to communicate over a low throughput segment.
For a given problem, some data is relevant, some data is not. Also, there are tunable lossy compression schemas. The models
presented here could be used to determine, on the fly, some compression parameters that will not alter the **data** in such
a way that the **information** retrieval is not hindered. This model must be tuned of course for the specific problem 
at hand. The criteria here is quite complex (do not decrease the information retrieval quality) and can't be directly
optimized. These networks can act as a proxy for the actual retrieval quality.
 
Head to the ``Installation`` section and then to ``src`` folder for a short explanation about the experiments.
 

## Installation:

Tested on Linux (Ubuntu 16.04+). Possibly runnable under Windows too. It relies on Anaconda environment.

Uses Python and PyTorch and fast dot ai v2. 

One must install fast dot ai in DEV mode. Version 2 is still being developed so things might change and present code will break. 
Create an issue and I'll sort it out eventually.

 * fastai2 is at commit bf455de9bc37c76f7f92b3c43227ef9d4779b614
 * fastcore is at commit 4a2d5ea702d0dc4a6c34c4acefafd9b494d9e222

Install CUDA 10.2: https://developer.nvidia.com/cuda-downloads

Create a Python Env and "compile" Jupyter lab support:

    conda create -y --copy -c conda-forge -c fastai -c pytorch -n f2 python\>=3.7 pytest numpy pandas matplotlib seaborn pillow scikit-learn scipy spacy pywavelets psutil pytest requests pyyaml jupyterlab ipywidgets ipympl nodejs\>=10  pytorch\>=1.3.0 torchvision\>=0.5 fastprogress\>=0.1.22 cudatoolkit 
    conda activate f2
    jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib

Fast dot AI specifics:

Head to a folder, clone and install the fastai2 and fastcore libraries. Make sure that the conda environment created above
is active! 

    git clone https://github.com/fastai/fastcore.git
    cd fastcore
    pip install -e ".[dev]"


    git clone https://github.com/fastai/fastai2
    cd fastai2
    pip install -e ".[dev]"

Now clone this repo and start the ``jupyter lab`` in ``src/`` folder.

 
