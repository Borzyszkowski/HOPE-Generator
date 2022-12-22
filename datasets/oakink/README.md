# OakInk
OakInk is a dataset containing 50.000 distinct affordance-aware and intent-oriented hand-object interactions.
In the scope of this work, we focus on its subset <em>OakInk shape</em> that contains 1800 3D object meshes and 
interactions using the MANO hand model.

## Download
To download the OakInk dataset, run the following script:
```Shell
./datasets/oakink/download_oakink.sh && export OAKINK_DIR=./_SOURCE_DATA/OakInk
```

## Load and Visualize

We provide a script for basic usage of data loading and visualizing. To visualize OakInk-Shape set with object category 
and subject intent
```Shell
python ./datasets/oakink/viz_oakink_shape.py --categories teapot --intent_mode use (--help)
```
