#!/bin/bash

# Set model and script base
MODEL_NAME="MLP 2D fingerprint"
SCRIPT_BASE="scripts/model/$MODEL_NAME"
PY_CMD="python $SCRIPT_BASE"

# Choose dataset
export DATA_NAME="random split"

# Paths to raw data
export RAW_DIR=$PWD/data/$DATA_NAME/raw
export TRAIN_FILE=$RAW_DIR/raw_train.csv
export VAL_FILE=$RAW_DIR/raw_val.csv
export TEST_FILE=$RAW_DIR/raw_test.csv

# Resources and output paths
export NUM_CORES=32
export PICKLE_DIR=$PWD/data/pickle
export PROCESSED_DATA_PATH=$PWD/data/$DATA_NAME/$MODEL_NAME/processed
export MODEL_PATH=$PWD/checkpoints/$MODEL_NAME/$DATA_NAME
export TEST_OUTPUT_PATH=$PWD/results/$MODEL_NAME/$DATA_NAME
export LOG_PATH=$PWD/logs/$MODEL_NAME/$DATA_NAME

# Validate existence of raw CSVs
[ -f $TRAIN_FILE ] || { echo "$TRAIN_FILE does not exist"; exit 1; }
[ -f $VAL_FILE ]   || { echo "$VAL_FILE does not exist"; exit 1; }
[ -f $TEST_FILE ]  || { echo "$TEST_FILE does not exist"; exit 1; }

# # Step 1: Preprocess the data (generate morganfp_*.pt)
$PY_CMD/01_preprocess.py \
    --raw_dir $RAW_DIR \
    --pickle_dir $PICKLE_DIR \
    --processed_dir $PROCESSED_DATA_PATH \
    --num_workers $NUM_CORES \
    --log_dir $LOG_PATH

# Step 2: Train the MLP model on Morgan fingerprints
$PY_CMD/02_train.py \
    --processed_dir $PROCESSED_DATA_PATH \
    --checkpoint_dir $MODEL_PATH \
    --log_dir $LOG_PATH \
    --num_epochs 200

# Step 3: Make predictions  
python $SCRIPT_BASE/03_predict.py \
    --processed_dir "$PROCESSED_DATA_PATH" \
    --checkpoint_dir "$MODEL_PATH" \
    --result_dir "$TEST_OUTPUT_PATH" \
    --log_dir "$LOG_PATH"