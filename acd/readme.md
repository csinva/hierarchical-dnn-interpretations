# source for calculating ACD interpretations

- [scores](scores) folder contains code for calculating different importance
  - `cd.py` file is the entry point for calculating CD scores
    - `cd_propagate.py` files contain code to calculate CD score across individual layers
    - `cd_architecture_specific.py` contains implementations for some specific architectures
  - `score_funcs.py` contains implementations of baselines and wrappers to return different scores    
- [agglomeration](agglomeration) folder contains code for aggregating scores to produce hierarchical interpretations
    - `agg_1d` is for text-like inputs and produces a sequence of 1-d components
    - `agg_2d` is for image-like inputs and produces a sequence of image segmentations
    - [util](util) scripts are used here for generating appropriately sized segments 
- there are a couple [tests](../tests) for some of this functionality as well

*note: most of the code is separated by 1d (for 1d inputs, such as text) and 2d (for 2d inputs, such as images)*