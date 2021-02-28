<h1 align="center"> Hierarchical neural-net interpretations (ACD) 🧠</h1>

<p align="center"> Produces hierarchical interpretations for a single prediction made by a pytorch neural network. Official code for <i>Hierarchical interpretations for neural network predictions</i> (ICLR 2019 <a href="https://openreview.net/pdf?id=SkEqro0ctQ">pdf</a>). </p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.6--3.8-blue">
  <img src="https://img.shields.io/badge/pytorch-1.0%2B-blue">
  <img src="https://img.shields.io/github/checks-status/csinva/hierarchical-dnn-interpretations/master">
  <img src="https://img.shields.io/pypi/v/acd?color=orange">
  <img src="https://static.pepy.tech/personalized-badge/acd?period=total&units=none&left_color=gray&right_color=orange&left_text=downloads">
</p>  
<p align="center">
	<a href="https://csinva.io/hierarchical-dnn-interpretations/">Documentation</a> •
  <a href="https://github.com/csinva/hierarchical-dnn-interpretations/tree/master/reproduce_figs">Demo notebooks</a>
</p>  
<p align="center">
	<i>Note: this repo is actively maintained. For any questions please file an issue.</i>
</p>


![](https://csinva.io/hierarchical-dnn-interpretations/intro.svg?sanitize=True)



# examples/documentation

- **installation**: `pip install acd` (or clone and run `python setup.py install`)
- **examples**: the [reproduce_figs](https://github.com/csinva/hierarchical-dnn-interpretations/tree/master/reproduce_figs) folder has notebooks with many demos
- **src**: the [acd](acd) folder contains the source for the method implementation
- allows for different types of interpretations by changing hyperparameters (explained in examples)
- all required data/models/code for reproducing are included in the [dsets](dsets) folder

| Inspecting NLP sentiment models    | Detecting adversarial examples      | Analyzing imagenet models           |
| ---------------------------------- | ----------------------------------- | ----------------------------------- |
| ![](reproduce_figs/figs/fig_2.png) | ![](reproduce_figs/figs/fig_s3.png) | ![](reproduce_figs/figs/fig_s2.png) |


# notes on using ACD on your own data
- the current CD implementation often works out-of-the box, especially for networks built on common layers, such as alexnet/vgg/resnet. However, if you have custom layers or layers not accessible in `net.modules()`, you may need to write a custom function to iterate through some layers of your network (for examples see `cd.py`). 
- to use baselines such build-up and occlusion, replace the pred_ims function by a function, which gets predictions from your model given a batch of examples.


# related work

- CDEP (ICML 2020 [pdf](https://arxiv.org/abs/1909.13584), [github](https://github.com/laura-rieger/deep-explanation-penalization)) - penalizes CD / ACD scores during training to make models generalize better
- TRIM (ICLR 2020 workshop [pdf](https://arxiv.org/abs/2003.01926), [github](https://github.com/csinva/transformation-importance)) - using simple reparameterizations, allows for calculating disentangled importances to transformations of the input (e.g. assigning importances to different frequencies)
- PDR framework (PNAS 2019 [pdf](https://arxiv.org/abs/1901.04592)) - an overarching framewwork for guiding and framing interpretable machine learning
- DAC (arXiv 2019 [pdf](https://arxiv.org/abs/1905.07631), [github](https://github.com/csinva/disentangled-attribution-curves)) - finds disentangled interpretations for random forests
- Baseline interpretability methods - the file `scores/score_funcs.py` also contains simple pytorch implementations of [integrated gradients](https://arxiv.org/abs/1703.01365) and the simple interpration technique `gradient * input`

# reference

- feel free to use/share this code openly
- if you find this code useful for your research, please cite the following:

 ```r
@inproceedings{
    singh2019hierarchical,
    title={Hierarchical interpretations for neural network predictions},
    author={Chandan Singh and W. James Murdoch and Bin Yu},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=SkEqro0ctQ},
}
 ```

