Code for using / reproducing ACD from the paper "Hierarchical interpretations for neural network predictions": <url here>. This code produces hierarchical interpretations for different types of trained neural networks in pytorch.

*note: this repo is still under development. For any questions please file an Issue*

# documentation
- also contains pytorch codes for interpretation baselines such as integrated gradients, occlusion
- allows for different types of interpretations by changing hyperparameters (explained in examples)
- tested with python3 and pytorch 0.3 on gpu
- see the reproduce_figs folder for notebooks with examples of using ACD to reproduce figures in the paper

# using ACD

- to use ACD on your own model, replace the models in the examples with your own trained models. Specifically, 3 things must be altered:
  1. the pred_ims function must be replaced by a function you write using your own trained model. This function gets predictions from a model given a batch of examples.
  2. the model must be replaced with your model
  3. the current cd implementation doesn't always work for all implementations. If you are getting an error inside of `cd.py`, you may need to write a custom function that iterates through the layers of your network (for examples see `cd.py`)