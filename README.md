# Acceleration Material Project: Drone Dataset Semantic Segmentation

## Problem tackled :

This repository contains implementation (model, training, evalutation) of both **Binary** and **Multiclass** problem
for the **Drone Semantic Segmentation challenge** in Deep Learning SICOM S9.

## Presentation of the project's architecture :

  - `/gricad/` : contains the output of the training : historic csv files, model dump, train curves and predictions.
  
  
  - `/code/` : contains all the python's code implementation.
  
    -  **Scripts** that can be executed with local terminal or on Gricad (provided that there is a `.h` file for the commands) 
       -  `main.py` : Run the training, validation and test of the chosen model, return the historic and the `.pt` saved model.
       -  `predict.py` : Given a test's images index python list, return the predictions of the chosen model.
       -  `get_curves.py` : Given `.csv` training historic files saved in an `output_path`, return the fitting curves (loss and metrics). 
  
    - **A module folder** `python/` required for the scripts execution :
      - `droneDataset.py` manages the PyTorch Dataset instance for the Drone Dataset and other data related functions
      - `metrics.py` contains the implementation of PA, MPA, IoU, mIoU.
      - `model.py`  models implementation : (UNet, SegFormer, UFormer)
      - `trainer.py` handle all the training, validation and testing functions.
      - `visualization.py` plot the figure, images and masks fancy display.
  
  
  - `config.yaml` : used to store configuration parameters such as training settings, model specifications, and file paths. 

## How to reproduce results ?

### On your local-personal environment :

#### Run a training :

  - Configure your desired list of parameters following the instructions provided in `config.yaml` file.
  /!\ Do not forget to disable distributed training /!\
  - In your terminal located at the source of the repository, type and enter : `python ./code/main.py`

#### Get predictions :

  - Once the training is finished, get some prediction visualization typing : `python ./code/predict.py`
    You can directly use this command to make prediction on a trained `.pt` model, just adapt the `output_path` parameter inside the
    `config.yaml` file to point toward the location of your model dump.

#### Get learning curves (loss, metrics) :

  - Finally get the training curves, using : `python ./code/get_curves.py`


### On Gricad Cluster :

#### Single GPU :

  - Create `.h` files adapted for the execution of each python scripts and run it, same instruction as for local run

#### Distributed Data Parallelism using GPUs :

/!\ Only available for training, script `main.py`. Then, predictions and learning curves are returned using a single GPU. /!\
/!\ Therefore, make sure to disable distributed `active` parameter once you get to the model prediction and learning curves generation /!\

  - Turn on `distributed::active=1` in the `config.yaml` file.
  - Adapt a `.h` file, it has to contains the following instructions for GPUs :
    - `export CUDA_VISIBLE_DEVICES=0,1,2,3`
    - `torchrun --nproc_per_node=4 code/main.py`
  - Run the `.h` file

Note : 
- It always use `localhost` master node adress because GriCad choose for ourself. 
- You may have to change the `port`, it is arbitrary but if someone is using the same as you, it will return an error.
- For the backend init method we realised all our computations using `ncc` since it is the one recommended for Nvidia GPU computations.
- Because of the version of pytorch you may have to install specific version of modules used to download certain models. 
