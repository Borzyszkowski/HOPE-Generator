# HOPE Generator: human-object interaction data synthesis

Hands are the primary means by which humans manipulate objects in the real-world, and measuring hand-object interactions (HOI) and hand-object pose estimation (HOPE) hold great potential for understanding human behavior. However, existing datasets are to small and lack comprehensive awareness of the object’s affordance and the hand’s interactions with it. For this reason, they are insufficient to elucidate fundamental principles of human movement.

In this project, we aim to create a simple machine learning pipeline to synthesize large-scale human-object interaction dataset that would help to gain better insights into the sensorimotor control in the long term. We apply novel machine learning techniques and develop our own algorithms to computationally generate new data. We propose to apply and refine deep learning algorithms to synthesize naturalistic movement.


## Installation

To be written by @Mirali

We assume that all the commands are executed using `./HOPE-Generator/` folder as a working directory.


## Datasets

We support three large-scale HOI datasets. To download them, perform the following steps:

- GRAB: `./datasets/grab/download_grab.sh`
- OakInk: `./datasets/oakink/download_oakink.sh`
- HOI4D (optional): Download HOI4D dataset through the instructions written in `./datasets/HOI4D`

## Body models

We use common body models such as SMPL-X and MANO. To download them, run the following script:

- `./body_models/download_body_models.sh`

## Download weights

We provide pre-trained weights for the neural networks. To download them, run the following script:

- `./models/download_models.sh`

## Generation

To run generation of HOI for 1800 unseen objects from the OakInk dataset, run the following script:

`python `./run_generation.py`

## Training (optional)

We allow the user to retrain the neural networks with custom parameters.
To train the models, run the following commands:
- GNet: `python ./train/GNet_train.py`
- MNet: `python ./train/MNet_train.py`

## Team:
* Antonino Scurria
* Bartlomiej Borzyszkowski
* Mirali Ahmadli
