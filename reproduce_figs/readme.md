**This folder contains notebooks to reproduce / extend the results in the paper.**

The [text notebook](text_fig2.ipynb) contains code to load a pretrained model on the SST dataset. Then, you can give it different sentences and observe the hierarchical interpretations it produces.

![](figs/fig_2.png)


# mnist

The [mnist notebook](mnist_figs3,s4.ipynb) contains code for analyzing the mnist dataset with ACD. Running this notebook will download the MNIST dataset, if you do not already have it.

- note: adversarial attacks require the `foolbox` and `randomgen` python packages (installable via pip)
  - 'boundary attack' is currently commented out as a result of an error in the `randomgen` package

![](figs/fig_s3.png)


# imagenet
The [imagenet notebook](imagenet_fig3,s1,s2.ipynb) contains code for using CD on CNN models. It comes with a pickle file containing a few imagenet images for testing out.

- note: redoing the imagenet results will be very slow if using cpu instead of gpu

![](figs/fig_s2.png)