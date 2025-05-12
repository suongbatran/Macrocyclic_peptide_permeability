#!/bin/bash

# Set environment variables
MODEL_NAME="baseline"
SCRIPT_BASE="scripts/model/$MODEL_NAME"
PY_CMD="python $SCRIPT_BASE"

# Change this tp "split_monomer" is you want to use other dataset
export DATA_NAME="split_random"

export TRAIN_FILE=$PWD/data/$DATA_NAME/raw/raw_train.csv
export VAL_FILE=$PWD/data/$DATA_NAME/raw/raw_val.csv
export TEST_FILE=$PWD/data/$DATA_NAME/raw/raw_test.csv
export NUM_CORES=32

export PROCESSED_DATA_PATH=$PWD/data/$DATA_NAME/$MODEL_NAME/processed
export MODEL_PATH=$PWD/checkpoints/$MODEL_NAME/$DATA_NAME
export TEST_OUTPUT_PATH=$PWD/results/$MODEL_NAME/$DATA_NAME
export LOG_PATH=$PWD/logs/$MODEL_NAME/$DATA_NAME

# Check if the required files exist
[ -f $TRAIN_FILE ] || { echo "$TRAIN_FILE does not exist"; exit 1; }
[ -f $VAL_FILE ] || { echo "$VAL_FILE does not exist"; exit 1; }
[ -f $TEST_FILE ] || { echo "$TEST_FILE does not exist"; exit 1; }

# Step 1: Preprocess the data
$PY_CMD/01_preprocess.py \
    --raw_dir $PWD/data/$DATA_NAME/raw \
    --pickle_dir $PWD/data/pickle \
    --processed_dir $PROCESSED_DATA_PATH \
    --num_workers $NUM_CORES \
    --log_dir $LOG_PATH

# Step 2: Train the model
$PY_CMD/02_train.py \
    --processed_dir $PROCESSED_DATA_PATH \
    --checkpoint_dir $MODEL_PATH \
    --log_dir $LOG_PATH \
    --num_epochs 200 \
    --learning_rate 1e-3

# Step 3: Make predictions
$PY_CMD/03_predict.py \
    --processed_dir $PROCESSED_DATA_PATH \
    --checkpoint_dir $MODEL_PATH \
    --result_dir $TEST_OUTPUT_PATH \
    --log_dir $LOG_PATH
