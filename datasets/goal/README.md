## Table of Contents
  * [Description](#description)
  * [Requirements](#requirements)
  * [Installation](#installation)
  * [Getting Started](#getting-started)
  * [Examples](#examples)
  * [Citation](#citation)
  * [License](#license)
  * [Acknowledgments](#acknowledgments)
  * [Contact](#contact)

## Description

This implementation:

- Can run GOAL on arbitrary objects provided by users (incl. computing on the fly the BPS representation for them).
- Provides a quick and easy demo on google colab to generate fullbody grasps by GNet for any given object (MNet results coming soon).
- Can run GOAL on the test objects of our dataset (with pre-computed object centering and BPS representation).
- Can retrain GNet and MNet, allowing users to change details in the training configuration.


## Requirements
This package has the following requirements:

* [Pytorch>=1.7.1](https://pytorch.org/get-started/locally/) 
* Python >=3.7.0
* [pytroch3d >=0.2.0](https://pytorch3d.org/) 
* [MANO](https://github.com/otaheri/MANO) 
* [SMPLX](https://github.com/vchoutas/smplx) 
* [bps_torch](https://github.com/otaheri/bps_torch) 
* [psbody-mesh](https://github.com/MPI-IS/mesh)

## Installation

To install the dependencies please follow the next steps:

- Clone this repository: 
    ```Shell
    git clone https://github.com/otaheri/GOAL
    cd GOAL
    ```
- Install the dependencies by the following command:
    ```
    pip install -r requirements.txt
    ```

## Getting started

For a quick demo of GNet you can give it a try on [google-colab here (Coming Soon)]().

Inorder to use the GOAL models please follow the steps below:


#### GNet and MNet data
- Download the GRAB dataset from the [GRAB website](https://grab.is.tue.mpg.de), and follow the instructions there to extract the files.
- Process the GNet data by running the command below.
```commandline
python data/process_gnet_data.py --grab-path /path/to/GRAB --smplx-path /path/to/smplx/models/
```
- Process the MNet data by running the command below.
```commandline
python data/process_mnet_data.py --grab-path /path/to/GRAB --smplx-path /path/to/smplx/models/
```

#### GNet and MNet models
- Please download the GNet and MNet models from our website and put them in the folders as below.
```bash
    GOAL
    ├── models
    │   │
    │   ├── GNet_model.pt
    │   ├── MNet_model.pt
    │   └── ...
    │   
    │
    .
    .
    .
```

#### SMPLX models
- Download body models following the steps on the [SMPLX repo](https://github.com/vchoutas/smplx) (skip this part if you already followed this for [GRAB dataset](https://github.com/otaheri/GRAB)).

### Generating Data

- #### Generate whole-body grasps and motions for test split.
    
    ```Shell
    python test/GOAL.py --work-dir /path/to/work/dir --grab-path /path/to/GRAB --smplx-path /path/to/models/
    ```
