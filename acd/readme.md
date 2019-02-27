**This folder contains the bulk of the code for ACD**

- the [scores](scores) folder contains code for getting importance scores for neural networks
  - importantly, the `cd.py` file contains code for getting CD scores
- the [agglomeration](agglomeration) folder contains code for aggregating scores to produce hierarchical interpretations
- note: most of the code is separated by 1d (for 1d inputs, such as text) and 2d (for 2d inputs, such as images)