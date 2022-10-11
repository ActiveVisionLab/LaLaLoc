# Overview

This is the code repository for LaLaLoc and LaLaLoc++.


* We currently provide:
  * Training and evaluation code for LaLaLoc, for both the Image-to-Layout and Layout-to-Layout configurations. 
  * Training and evaluation code for LaLaLoc++'s plan and image branches.
  * Pretrained models for all the provided configs.

## LaLaLoc++: Global Floor Plan Comprehension for Layout Localisation in Unvisited Environments
**Henry Howard-Jenkins and Victor Adrian Prisacariu**
**(ECCV 2022)**

[Project Page](https://lalalocpp.active.vision) | Paper(coming soon!)

![LaLaLoc++ Overview](assets/lll++_overview.png)

## LaLaLoc: Latent Layout Localisation in Dynamic, Unvisited Environments
**Henry Howard-Jenkins, Jose-Raul Ruiz-Sarmiento and Victor Adrian Prisacariu**
**(ICCV 2021)**

[Project Page](https://lalaloc.active.vision) | [Paper](https://arxiv.org/abs/2104.09169)

![LaLaLoc Overview](assets/overview.png)


# Setup
## Installing Requirements

* Create conda environment:
```
conda create -n lalaloc python==3.8
conda activate lalaloc
```
* Install PyTorch:
```
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
```
* Install Pytorch Lightning:
```
conda install -c conda-forge pytorch-lightning==1.1.5
```
* Install Pytorch3d:
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install -c pytorch3d pytorch3d==0.4.0 
```
* Install Pymesh
    * Follow build and install instructions: https://github.com/PyMesh/PyMesh
* Install Redner and OpenCV:
```
pip install redner-gpu opencv-python
```
* Install Scikit-Learn:
```
conda install -c anaconda scikit-learn
```

## Download the Structured3D Dataset
* Information provided here: https://github.com/bertjiazheng/Structured3D

# Usage
### Layout/Plan Branch
* Train LaLaLoc's layout branch or LaLaLoc++'s plan branch.
```
# LaLaLoc layout branch
python train.py -c configs/layout_branch.yaml \
    DATASET.PATH [path/to/dataset]
```
```
# LaLaLoc++ plan branch
python train.py -c configs/lalaloc_pp/plan_branch.yaml \
    DATASET.PATH [path/to/dataset]
```
* Test LaLaLoc's layout branch:
    * Perform evaluation of the trained layout branch on a sampled grid of 0.5m with VDR and LPO.
    
    Note: Testing LaLaLoc++'s plan branch isn't particularly meaningful.
```
python train.py -c configs/layout_branch.yaml -t [path/to/checkpoint] \
    DATASET.PATH [path/to/dataset] \
    SYSTEM.NUM_GPUS 1 \
    TEST.VOGEL_DISC_REFINE True \
    TEST.LATENT_POSE_OPTIMISATION True \
    TEST.POSE_SAMPLE_STEP 500
```

### Image Branch
* Train the image branch for LaLaLoc and LaLaLoc++
    * Perform training of the image branch with the layout/plan branch from a previous training run.
```
# LaLaLoc image branch
python train.py -c configs/image_branch.yaml \
    DATASET.PATH [path/to/dataset] \
    TRAIN.SOURCE_WEIGHTS [path/to/layout_branch_checkpoint]
```
```
# LaLaLoc++ image branch
python train.py -c configs/lalaloc_pp/image_branch.yaml \
    DATASET.PATH [path/to/dataset] \
    TRAIN.SOURCE_WEIGHTS [path/to/plan_branch_checkpoint]
```

* Test image branch
```
# LaLaLoc image branch
python train.py -c configs/image_branch.yaml -t [path/to/checkpoint] \
    DATASET.PATH [path/to/dataset] \
    SYSTEM.NUM_GPUS 1 \
    TEST.VOGEL_DISC_REFINE True \
    TEST.LATENT_POSE_OPTIMISATION True \
    TEST.POSE_SAMPLE_STEP 500
```
```
# LaLaLoc++ image branch
python train.py -c configs/lalaloc_pp/transfomer_image_branch.yaml -t [path/to/checkpoint] \
    DATASET.PATH [path/to/dataset] \
    SYSTEM.NUM_GPUS 1 \
```

# Citations
```
@article{howard2022lalaloc++,
  title={LaLaLoc++: Global Floor Plan Comprehension for Layout Localisation in Unvisited Environments},
  author={Howard-Jenkins, Henry and Prisacariu, Victor Adrian},
  booktitle={Proceedings of the European Conference on Computer Vision},
  pages={},
  year={2022}
}
```
```
@inproceedings{howard2021lalaloc,
  title={Lalaloc: Latent layout localisation in dynamic, unvisited environments},
  author={Howard-Jenkins, Henry and Ruiz-Sarmiento, Jose-Raul and Prisacariu, Victor Adrian},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10107--10116},
  year={2021}
}
```