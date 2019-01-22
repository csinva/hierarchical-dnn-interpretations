Code for using / reproducing ACD from the paper "[Hierarchical interpretations for neural network predictions](https://openreview.net/pdf?id=SkEqro0ctQ)" (ICLR 2019). This code produces hierarchical interpretations for different types of trained neural networks in pytorch.

*Note: this repo is actively maintained. For any questions please file an issue.*

# documentation
- also contains pytorch codes for interpretation baselines such as integrated gradients, occlusion
- allows for different types of interpretations by changing hyperparameters (explained in examples)
- tested with python3 and pytorch with/without gpu
- see the reproduce_figs folder for notebooks with examples of using ACD to reproduce figures in the paper

# using ACD

- to use ACD on your own model, replace the models in the examples with your own trained models. Specifically, 3 things must be altered:
  1. the pred_ims function must be replaced by a function you write using your own trained model. This function gets predictions from a model given a batch of examples.
  2. the model must be replaced with your model
  3. the current CD implementation doesn't always work for all types of networks. If you are getting an error inside of `cd.py`, you may need to write a custom function that iterates through the layers of your network (for examples see `cd.py`)

# related work

- this work is part of an overarching project on interpretable machine learning, guided by the [PDR framework](https://arxiv.org/abs/1901.04592)
- for related work, see the [github repo](https://github.com/jamie-murdoch/ContextualDecomposition) for contextual decomposition ([ICLR 2018](https://openreview.net/pdf?id=rkRwGg-0Z))