## Supporting data for "Inductive biases and Self Supervised Learning in modelling a physical heating system"

The data used in the experiments is from a household. We collected many features along few years. Considerable information
about the habits can be extracted from the data. This is a strong privacy violation, so we can't just release the data directly.

However, in recent years there was (and still is) an abundance of papers that are hard (or impossible) to reproduce. Code 
is made available, experimental setup is presented as used in the paper, but the data was truncated, sampled and slightly 
altered (ex: missing data imputation and other minor tweaks) so we can achieve some degree of anonymity and reduce the 
risk of doxxing.

There are only few days of data exported. The data is more densely sampled, so the volumes might appear larger than those
used in the paper but in reality there is not much diversity. As a result, the models presented in the paper overfit.

Trimming some of the capacity yielded two networks that are able to "beat" the *Zero* predictor and these networks demonstrate
how to load, train and predict with this dataset.

If there are some opportunities to perform experiments on another physical system, a system that does not raise privacy issues,
the data will be made available!

In the meantime, I hope that this small sample is enough to stir some curiosity and to demonstrate the usefulness of the
methods developed here.

 * ``train_samples.npy``  -  Training data subsample    -  7255 samples, each sample have 160 timestamps and 4 features
 * ``valid_samples.npy`` -  Validation data subsample  -  1073 samples, each sample have 160 timestamps and 4 features

Note that the numpy strucutre (sample x time x feature) is different than the network inputs (batch x feature x time). 
The reshaping is handled by ``network_definitions.FeatureItemizer``.

All the data in ``valid_samples`` is *in the future* of ``train_samples``. 
