# Evaluating models for Action Recognition on EPIC-KITCHENS-100

The code in this repo is a clone from [https://github.com/epic-kitchens/epic-kitchens-slowfast](https://github.com/epic-kitchens/epic-kitchens-slowfast) and adapted to evaluate more models on the EPIC-KITCHENS-100 dataset.

Work in progress! 


You can currently run different types of inference with the following pre-trained models:

- SlowFast
- Omnivore
- TSM

Inference can be run on these datasets

- EPIC-KITCHENS 100 (validation set)

The types of inference supported are:

- Original SlowFast inference
- Original Omnivore inference
- Sliding fixed-size window:
    - strictly inside action boundaries
    - inside and around action boundaries
    - sliding over full videos

## Preparation

- Please install all the requirements found in the original SlowFast repo ([link](https://github.com/facebookresearch/SlowFast/blob/master/INSTALL.md))
* Add this repository to $PYTHONPATH.
```
export PYTHONPATH=/path/to/epic-kitchens-slowfast/slowfast:$PYTHONPATH
```
* From the annotation repository of EPIC-KITCHENS-100 ([link](https://github.com/epic-kitchens/epic-kitchens-100-annotations)), download: EPIC_100_train.pkl, EPIC_100_validation.pkl, and EPIC_100_test_timestamps.pkl. EPIC_100_train.pkl and EPIC_100_validation.pkl will be used for training/validation, while EPIC_100_test_timestamps.pkl will be used to obtain the scores to submit in the AR challenge.
* Download only the RGB frames of EPIC-KITCHENS-100 dataset using the download scripts found [here](https://github.com/epic-kitchens/epic-kitchens-download-scripts). 
The training/validation code expects the following folder structure for the dataset:
```
├── dataset_root
|   ├── P01
|   |   ├── rgb_frames
|   |   |   |    ├── P01_01
|   |   |   |    |    ├── frame_0000000000.jpg
|   |   |   |    |    ├── frame_0000000001.jpg
|   |   |   |    |    ├── .
|   |   |   |    |    ├── .
|   |   |   |    |    ├── .
|   |   |   |    .    
|   |   |   |    .    
|   |   |   |    .
|   ├── .
|   ├── .
|   ├── .
|   ├── P37
|   |   ├── rgb_frames
|   |   |   |    ├── P37_101
|   |   |   |    |    ├── frame_0000000000.jpg
|   |   |   |    |    ├── frame_0000000001.jpg
|   |   |   |    |    ├── .
|   |   |   |    |    ├── .
|   |   |   |    |    ├── .
|   |   |   |    .    
|   |   |   |    .    
|   |   |   |    .
```
So, after downloading the dataset navigate under <participant_id>/rgb_frames for each participant and untar each video's frames in its corresponding folder, e.g for P01_01.tar you should create a folder P01_01 and extract the contents of the tar file inside.

## To run
To obtain scores on the validation set (using the model trained on the concatenation of the training and validation sets) run:
```
python tools/run_net.py --cfg configs/EPIC-KITCHENS/<configi*file*name>.yaml  
EPICKITCHENS.VISUAL_DATA_DIR /path/to/dataset 
EPICKITCHENS.ANNOTATIONS_DIR /path/to/annotations 
```

## License 

The code is published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, found [here](https://creativecommons.org/licenses/by-nc-sa/4.0/).
