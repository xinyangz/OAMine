GPU=0,1
BERT=/path/to/bert/checkpoint
DATA_DIR=/path/to/data
OUTPUT_DIR=/path/to/output

CUDA_VISIBLE_DEVICES=$GPU python launch_preprocess.py \
 --bert $BERT \
 --data_dir $DATA_DIR \
 --output_dir $OUTPUT_DIR \
 --cuda

