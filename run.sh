# MTCN Audio-Visual

python tools/run_net.py \
  --cfg configs/EPIC-KITCHENS/MTCN.yaml \
  OUTPUT_DIR ./output/mtcn \
  EPICKITCHENS.VISUAL_DATA_DIR /datasets/EPIC-KITCHENS \
  TRAIN.ENABLE False TEST.ENABLE True \
  TEST.CHECKPOINT_FILE_PATHS.audio_encoder checkpoints/AUD_SLOWFAST_EPIC.pyth \
  TEST.CHECKPOINT_FILE_PATHS.video_encoder checkpoints/SlowFast.pyth \
  TEST.CHECKPOINT_FILE_PATHS.cross_attention checkpoints/mtcn_av_sf_epic-kitchens-100.pyth

# MTCN Visual Only

python tools/run_net.py \
  --cfg configs/EPIC-KITCHENS/MTCN.yaml \
  OUTPUT_DIR ./output/mtcn \
  EPICKITCHENS.VISUAL_DATA_DIR /datasets/EPIC-KITCHENS \
  TRAIN.ENABLE False TEST.ENABLE True \
  MODEL.AUDIO False \
  TEST.CHECKPOINT_FILE_PATHS.video_encoder checkpoints/SlowFast.pyth \
  TEST.CHECKPOINT_FILE_PATHS.cross_attention checkpoints/mtcn_av_sf_epic-kitchens-100.pyth

# Audio SlowFast

python tools/run_net.py \
  --cfg configs/EPIC-KITCHENS/AUDIO_SLOWFAST_R50.yaml \
  OUTPUT_DIR ./output/audslowfast \
  EPICKITCHENS.VISUAL_DATA_DIR /datasets/EPIC-KITCHENS \
  TRAIN.ENABLE False TEST.ENABLE True \
  TEST.CHECKPOINT_FILE_PATH checkpoints/AUD_SLOWFAST_EPIC.pyth

python tools/run_net.py \
  --cfg configs/EPIC-KITCHENS/AUDIO_SLOWFAST_R50.yaml \
  OUTPUT_DIR ./output/audslowfast \
  EPICKITCHENS.AUDIO_DATA_FILE ../EPIC_audio.hdf5 \
  TRAIN.ENABLE False TEST.ENABLE True \
  TEST.CHECKPOINT_FILE_PATH checkpoints/AUD_SLOWFAST_EPIC.pyth

# Omnivore

python tools/run_net.py \
  --cfg configs/EPIC-KITCHENS/OMNIVORE.yaml \
  OUTPUT_DIR ./output/omnivore \
  EPICKITCHENS.VISUAL_DATA_DIR /datasets/EPIC-KITCHENS \
  TRAIN.ENABLE False TEST.ENABLE True

# Slowfast

python tools/run_net.py \
  --cfg configs/EPIC-KITCHENS/SLOWFAST_8x8_R50.yaml \
  OUTPUT_DIR ./output/slowfast \
  EPICKITCHENS.VISUAL_DATA_DIR /datasets/EPIC-KITCHENS \
  TRAIN.ENABLE False TEST.ENABLE True \
  TEST.CHECKPOINT_FILE_PATH checkpoints/SlowFast.pyth

# Audio SlowFast in original repo

python tools/run_net.py \
    --cfg configs/EPIC-KITCHENS/SLOWFAST_R50.yaml NUM_GPUS 1 \
    OUTPUT_DIR ./output \
    EPICKITCHENS.AUDIO_DATA_FILE ../EPIC_audio.hdf5 \
    EPICKITCHENS.ANNOTATIONS_DIR ../epic-kitchens-100-annotations \
    TRAIN.ENABLE False TEST.ENABLE True \
    TEST.CHECKPOINT_FILE_PATH ./checkpoints/SLOWFAST_EPIC.pyth
