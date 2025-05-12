#!/bin/bash

# Set environment variables
MODEL_NAME="Graph_2d"
SCRIPT_BASE="scripts/model/$MODEL_NAME"
PY_CMD="python $SCRIPT_BASE"

# Change this tp "split_monomer" is you want to use other dataset
export DATA_NAME="split_monomer"

export TRAIN_FILE=$PWD/data/$DATA_NAME/raw/raw_train.csv
export VAL_FILE=$PWD/data/$DATA_NAME/raw/raw_val.csv
export TEST_FILE=$PWD/data/$DATA_NAME/raw/raw_test.csv
export NUM_CORES=16

export PROCESSED_DATA_PATH=$PWD/data/$DATA_NAME/$MODEL_NAME/processed
export MODEL_PATH=$PWD/checkpoints/$MODEL_NAME/$DATA_NAME
export TEST_OUTPUT_PATH=$PWD/results/$MODEL_NAME/$DATA_NAME
export LOG_PATH=$PWD/logs/$MODEL_NAME/$DATA_NAME

# Check if the required files exist
[ -f $TRAIN_FILE ] || { echo "$TRAIN_FILE does not exist"; exit 1; }
[ -f $VAL_FILE ] || { echo "$VAL_FILE does not exist"; exit 1; }
[ -f $TEST_FILE ] || { echo "$TEST_FILE does not exist"; exit 1; }

# Hyperparameters
EPOCHS=200                      # Lightning max_epochs
PATIENCE=30                     # early-stop patience

# Comma-lists below trigger grid-search; leave a single value for one run
HIDDEN="64,128,256"                 # hidden dims to sweep
LAYERS="3,4,5,6"                    # GNN layers to sweep
LR="1e-3,5e-4,5e-3"                  # learning rates to sweep
BATCH_SIZE=32   

# Step 1: Preprocess the data
$PY_CMD/01_preprocess.py \
    --raw_dir $PWD/data/$DATA_NAME/raw \
    --pickle_dir $PWD/data/pickle \
    --processed_dir $PROCESSED_DATA_PATH \
    --num_workers $NUM_CORES \
    --log_dir $LOG_PATH

# Step 2: Train the model
$PY_CMD/train.py \
    --processed_dir  "$PROCESSED_DATA_PATH" \
    --checkpoint_dir "$MODEL_PATH" \
    --log_dir        "$LOG_PATH" \
    --hidden         "$HIDDEN" \
    --layers         "$LAYERS" \
    --batch_size     "$BATCH_SIZE" \
    --lr             "$LR" \
    --epochs         "$EPOCHS" \
    --patience       "$PATIENCE"

# Step 3: Make predictions
$PY_CMD/03_predict.py \
    --processed_dir $PROCESSED_DATA_PATH \
    --checkpoint_dir $MODEL_PATH \
    --result_dir $TEST_OUTPUT_PATH \
    --log_dir $LOG_PATH
