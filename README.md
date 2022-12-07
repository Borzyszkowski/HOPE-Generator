# HOPE Generator: human-object interaction data synthesis

Hands are the primary means by which humans manipulate objects in the real-world, and measuring hand-object interactions (HOI) and hand-object pose estimation (HOPE) hold great potential for understanding human behavior. However, existing datasets are to small and lack comprehensive awareness of the object’s affordance and the hand’s interactions with it. For this reason, they are insufficient to elucidate fundamental principles of human movement.

In this project, we aim to create a simple machine learning pipeline to synthesize large-scale human-object interaction dataset that would help to gain better insights into the sensorimotor control in the long term. We apply novel machine learning techniques and develop our own algorithms to computationally generate new data. We propose to apply and refine deep learning algorithms to synthesize naturalistic movement.


## Environment

To create the environment, run the following command:

`conda env create -f environment.yml`

## Datasets

Download  Grab dataset using bash script:

- GRAB: `./datasets/grab/download_grab.sh`

- OakInk: Download OakInk dataset through the instructions written in `HOPE-generator/datasets/OakInk`

## Body models

Download body models using a script:

`./body_models/download_body_models.sh`

## Training

To train the model, run the following command:

`python run.py --config <path_to_config_file> --mode train --agent <train_agent_name>`

## Testing

To test the model, run the following command:

`python run.py --config <path_to_config_file> --mode test --agent <test_agent_name>`


## Team:
* Antonino Scurria
* Bartlomiej Borzyszkowski
* Mirali Ahmadli
