# HOPE Generator: human-object interaction data synthesis

Hands are the primary means by which humans manipulate objects in the real-world, and measuring hand-object interactions (HOI) and hand-object pose estimation (HOPE) hold great potential for understanding human behavior. However, existing datasets are to small and lack comprehensive awareness of the object’s affordance and the hand’s interactions with it. For this reason, they are insufficient to elucidate fundamental principles of human movement.

In this project, we aim to create a simple machine learning pipeline to synthesize large-scale human-object interaction dataset that would help to gain better insights into the sensorimotor control in the long term. We apply novel machine learning techniques and develop our own algorithms to computationally generate new data. We propose to apply and refine deep learning algorithms to synthesize naturalistic movement.


## Installation

To be written by @Mirali

###### We assume that all the commands are executed from the `./HOPE-Generator` as a working directory.


## Datasets

We support three large-scale HOI datasets. To download them, perform the following steps:

- GRAB: `./datasets/grab/download_grab.sh`
- OakInk: `./datasets/oakink/download_oakink.sh && export OAKINK_DIR=./_SOURCE_DATA/OakInk`
- HOI4D (optional): Download HOI4D dataset through the instructions given in `./datasets/HOI4D`

## Body models

We use common body models such as SMPL-X and MANO. To download them, run the following script:

- `./body_models/download_body_models.sh`

## Pre-trained weights

We provide pre-trained weights for the neural networks. To download them, run the following script:

- `./models/download_models.sh`

## Data preprocessing


## Generation

To use pre-trained weights run generation of HOI for 1800 unseen objects from the OakInk dataset, run the following script:

- `python ./run_generation.py`


## Results (optional)

Because large-scale HOI generation is time-consuming, we provide our results for 100 sequences. To download them, run the following script:
- `./download_results.sh`

It will generate two folders:
- visualizations: `./_RESULTS/Downloaded/100_objects_meshes/`
- sequences with 3D meshes: `./_RESULTS/Downloaded/100_objects_visualized/`

## Training (optional)

We allow the user to retrain the neural networks with custom parameters.
To train the models from scratch, run the following commands:
- GNet: `python ./train/GNet_train.py`
- MNet: `python ./train/MNet_train.py`

## Team:
* Antonino Scurria
* Bartlomiej Borzyszkowski
* Mirali Ahmadli
