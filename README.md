# GFESF-Grasp
This repository contains the official implementation of GFESF-Grasp from the paper:

**Global Feature Enhancement and Skip-Connected Fusion for Grasping Detection**

Shengjun Xu, Xiaoyi Wang, Rui Shen, Ya shi, Bohan Zhan, Erhu Liu and Xiaohan Li.


Please clone this GitHub repo before proceeding with the installation.

```bash
git clone https://github.com/wangxauat/gfesf.git
```

## Installation using Anaconda

The code was tested on Python 3.8.10 and PyTorch 1.12 (CUDA 11.3). NVIDIA GPUs are needed for both training and testing.

1. Create a new conda environment
    
    ```bash
    conda create --name gefsf python=3.8.10
    ```
    
2. Install PyTorch 1.12 for CUDA 11.3
    
    ```bash
    conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

    ```
    
3. Install the required Python packages
    
    ```bash
    pip install -r requirements.txt
    ```
    
    

## Dataset Preparation

1. Download and extract the [Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp).
2. Download and extract the [Jacquard Dataset](https://jacquard.liris.cnrs.fr/files/database/download.php).
3. For the Cornell and Jacquard dataset, the folders containing the images and labels should be arranged in the following manner:
    
    ```
    GFESF-Grasp
    | - - data
     `- - | - - cornell
          |  `- - | - - 01
          |       | - - 02
          |       | - - 03
          |       | - - 04
          |       | - - 05
          |       | - - 06
          |       | - - 07
          |       | - - 08
          |       | - - 09
          |       | - - 10
          |        ` - - backgrounds
          ` - - jacquard
            ` - - | - - Jacquard_Dataset_0
                  | - - Jacquard_Dataset_1
                  | - - Jacquard_Dataset_2
                  | - - Jacquard_Dataset_3
                  | - - Jacquard_Dataset_4
                  | - - Jacquard_Dataset_5
                  | - - Jacquard_Dataset_6
                  | - - Jacquard_Dataset_7
                  | - - Jacquard_Dataset_8
                  | - - Jacquard_Dataset_9
                  | - - Jacquard_Dataset_10
                   ` - - Jacquard_Dataset_11
    ```
    
4. For the Cornell Grasping Dataset. convert the PCD files (pcdXXXX.txt) to depth images by running
    
    ```bash
    python -m utils.dataset_preprocessing.generate_cornell_depth data/cornell
    ```
    

## Training

Run `train.py --help` to see the full list of options and description for each option.

.A basic example would be:

```bash
python train.py --description <write a description> --network mynet --dataset cornell --dataset-path data/cornell --use-rgb 1 --use-depth 1
```

Some important flags are:

- `--dataset` to select the dataset you want to use for training.
- `--dataset-path` to provide the path to the selected dataset.
- `--input-size` to change the size of the input image. **Note that the input image must be a multiple of 8**
- `--use-rgb` to use RGB images during training. Set 1 for true and 0 for false.
- `--use-depth` to use depth images during training.  Set 1 for true and 0 for false.



The trained models will be stored in the `output/models` directory. 

## Evaluation/Visualization

Run `evaluate.py ` to see the full list of options and description for each option.

Some important flags are:

- `--iou-eval` to evaluate using the IoU between grasping rectangles metric
- `--jacquard-ouptut` to generate output files in the format required for simulated testing against the Jacquard dataset.
- `--vis` to plot the network output and predicted grasp rectangles

A basic example would be:

```bash
python eval.py --network <path to trained network> --dataset jacquard --dataset-path data/jacquard --jacquard-output --iou-eval
```


```