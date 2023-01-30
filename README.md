# Evaluating models for Action Recognition on EPIC-KITCHENS-100

The code in this repo is a clone from [https://github.com/epic-kitchens/epic-kitchens-slowfast](https://github.com/epic-kitchens/epic-kitchens-slowfast) and adapted to evaluate more models on the EPIC-KITCHENS-100 dataset.

Work in progress! 


You can currently run different types of inference with the following pre-trained models:

- SlowFast
- Omnivore

Inference can be run on these datasets

- EPIC-KITCHENS 100 (validation set)

## Models
The models used in this project:
 - [SlowFast](https://github.com/epic-kitchens/epic-kitchens-slowfast)
 - [Auditory SlowFast](https://github.com/ekazakos/auditory-slow-fast)
 - [Omnivore](https://github.com/facebookresearch/omnivore)
 - [MTCN](https://github.com/ekazakos/MTCN)
 - [TSM]()

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
python tools/run_net.py \
  --cfg configs/EPIC-KITCHENS/SLOWFAST_8x8_R50.yaml \
  OUTPUT_DIR /path/to/output_dir \
  EPICKITCHENS.VISUAL_DATA_DIR /path/to/dataset \
  EPICKITCHENS.ANNOTATIONS_DIR /path/to/annotations \
  TRAIN.CHECKPOINT_FILE_PATH /path/to/SLOWFAST_8x8_R50.pkl
```
To validate the model run:
```
python tools/run_net.py \
  --cfg configs/EPIC-KITCHENS/SLOWFAST_8x8_R50.yaml \
  OUTPUT_DIR /path/to/experiment_dir \
  EPICKITCHENS.VISUAL_DATA_DIR /path/to/dataset \
  EPICKITCHENS.ANNOTATIONS_DIR /path/to/annotations \
  TRAIN.ENABLE False TEST.ENABLE True \
  TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth
```
After tuning the model's hyperparams using the validation set, we train the model that will be used for obtaining the test set's scores on the concatenation of the training and validation sets. To train the model on the concatenation of the training and validation sets run:
```
python tools/run_net.py \
  --cfg configs/EPIC-KITCHENS/SLOWFAST_8x8_R50.yaml \
  OUTPUT_DIR /path/to/output_dir \
  EPICKITCHENS.VISUAL_DATA_DIR /path/to/dataset \
  EPICKITCHENS.ANNOTATIONS_DIR /path/to/annotations \
  EPICKITCHENS.TRAIN_PLUS_VAL True \
  TRAIN.CHECKPOINT_FILE_PATH /path/to/SLOWFAST_8x8_R50.pkl
```
To obtain scores on the test set (using the model trained on the concatenation of the training and validation sets) run:
```
python tools/run_net.py \
  --cfg configs/EPIC-KITCHENS/SLOWFAST_8x8_R50.yaml \
  OUTPUT_DIR /path/to/experiment_dir \
  EPICKITCHENS.VISUAL_DATA_DIR /path/to/dataset \
  EPICKITCHENS.ANNOTATIONS_DIR /path/to/annotations \
  TRAIN.ENABLE False TEST.ENABLE True \
  TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth \
  EPICKITCHENS.TEST_LIST EPIC_100_test_timestamps.pkl \
  EPICKITCHENS.TEST_SPLIT test

# ---
python tools/run_net.py --cfg configs/EPIC-KITCHENS/<config file name>.yaml  
EPICKITCHENS.VISUAL_DATA_DIR /path/to/dataset 
EPICKITCHENS.ANNOTATIONS_DIR /path/to/annotations 
EPICKITCHENS.TEST_LIST EPIC_100_test_timestamps.pkl
```

## License 

The code is published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, found [here](https://creativecommons.org/licenses/by-nc-sa/4.0/).
