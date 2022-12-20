# HOPE Generator: human-object interaction data synthesis

Hands are the primary means by which humans manipulate objects in the real-world, and measuring hand-object interactions (HOI) and hand-object pose estimation (HOPE) hold great potential for understanding human behavior. However, existing datasets are to small and lack comprehensive awareness of the object’s affordance and the hand’s interactions with it. For this reason, they are insufficient to elucidate fundamental principles of human movement.

In this project, we aim to create a simple machine learning pipeline to synthesize large-scale human-object interaction dataset that would help to gain better insights into the sensorimotor control in the long term. We apply novel machine learning techniques and develop our own algorithms to computationally generate new data. We propose to apply and refine deep learning algorithms to synthesize naturalistic movement.

![Alt text](img/img_1.png?raw=true "Title")
  
###### Remark: We assume that all the commands below are executed from the `./HOPE-Generator` as a working directory.
## Installation



The core of HOPE Generator is based on [GOAL](https://arxiv.org/pdf/2112.11454.pdf), therefore it requires to install its all dependencies.


###### To be written by @Mirali




## Datasets

We support three large-scale HOI datasets. To download them, perform the following steps:

- [GRAB](https://arxiv.org/pdf/2008.11200.pdf): `./datasets/grab/download_grab.sh`
- [OakInk](https://arxiv.org/pdf/2203.15709.pdf): `./datasets/oakink/download_oakink.sh && export OAKINK_DIR=./_SOURCE_DATA/OakInk`
- [HOI4D](https://arxiv.org/pdf/2203.01577.pdf) <em>(optional)</em>: Download HOI4D dataset using the instructions given in `./datasets/HOI4D`

To learn more about the datasets, i.e. to visualize them, follow the README.md files in their corresponding directories.

## Body models

We use common body models such as [SMPL-X](https://ps.is.mpg.de/uploads_file/attachment/attachment/497/SMPL-X.pdf) and [MANO](https://ps.is.mpg.de/uploads_file/attachment/attachment/392/Embodied_Hands_SiggraphAsia2017.pdf). To download them, run the following script:
```Shell
./body_models/download_body_models.sh
```
## Pre-trained weights

We provide pre-trained weights for the neural networks. To download them, run the following script:
```Shell
./models/download_models.sh
```
## Data preprocessing

Prepare data for GNet (grasp generation) and MNet (motion generation): 
```Shell
python ./data_preparation/process_data_gnet.py
python ./data_preparation/process_data_mnet.py
```
## Generation

After performing the above steps, your project should have the following structure:
  ```
  HOPE-Generator
  ├── _BODY_MODELS
  │   ├── model_correspondences
  │   └── models
  ├── _DATA
  │   ├── GNet_data
  │   └── MNet_data
  ├── _SOURCE_DATA
  │   ├── GRAB
  │   └── OakInk
  ├── models
  │   ├── GNet_model.py
  │   ├── MNet_model.py
  │   └── ...
  ├── ...
  ├── ...
  └── run_generation.py
  ```

Next, you can use pre-trained weights to generate HOI for unseen objects.

For 5 test objects from the GRAB dataset run: 
```Shell
python ./run_generation.py --dataset-choice GRAB
```
For 1800 objects from the OakInk dataset run: 
```Shell
python ./run_generation.py --dataset-choice OakInk
```
## Results (optional)

Because large-scale HOI generation is time-consuming, we provide our results for 100 sequences as a reference. To download them, run the following script:
```Shell
./download_results.sh`
```

It will generate two folders with the results that contain static whole-body grasps as well as sequences of motion:
- Visualizations: `./_RESULTS/Downloaded/100_objects_meshes/`
- Sequences with 3D meshes: `./_RESULTS/Downloaded/100_objects_visualized/`

Alternatively, one can download an example interaction directly from GitHub and open it as an <em>.html</em> file in the browser:
- Motion: `img/s5_C90001_1_motion.html`
- Static grasp: `img/s5_C91001_1_grasp.html`

## Training (optional)

We allow the user to retrain the neural networks with custom parameters.
To train the models from scratch, run the following commands:
- GNet: `python ./train/GNet_train.py`
- MNet: `python ./train/MNet_train.py`

## Authors:
* Antonino Scurria [antonino.scurria@epfl.ch]
* Bartlomiej Borzyszkowski [bartlomiej.borzyszkowski@epfl.ch]
* Mirali Ahmadli [mirali.ahmadli@epfl.ch]
