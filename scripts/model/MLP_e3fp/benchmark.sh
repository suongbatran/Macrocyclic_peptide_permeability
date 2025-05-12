#!/bin/bash

# Set model and script base
MODEL_NAME="MLP_e3fp"
SCRIPT_BASE="scripts/model/$MODEL_NAME"

# Choose dataset
export DATA_NAME="split_random"

# Paths to raw data
export RAW_DIR=$PWD/data/$DATA_NAME/raw
export TRAIN_FILE=$RAW_DIR/raw_train.csv
export VAL_FILE=$RAW_DIR/raw_val.csv
export TEST_FILE=$RAW_DIR/raw_test.csv

mkdir -p "$PWD/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PWD/logs/${MODEL_NAME}/${DATA_NAME}/run_${TIMESTAMP}.txt"

# Start logging
echo "Starting hyperparameter search at $(date)" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"

# Resources and output paths
export NUM_CORES=32
export PICKLE_DIR=$PWD/data/pickle
export BATCH_SIZE=10
export EPOCHS=100
export PATIENCE=20

# Validate existence of raw CSVs
[ -f $TRAIN_FILE ] || { echo "$TRAIN_FILE does not exist"; exit 1; }
[ -f $VAL_FILE ]   || { echo "$VAL_FILE does not exist"; exit 1; }
[ -f $TEST_FILE ]  || { echo "$TEST_FILE does not exist"; exit 1; }

HIDDEN_LIST=(64 128 256)
BITS_LIST=(1024 2048) 
LEVEL_LIST=(3 4 5)
RADIUS_LIST=(1.0 2.0 3.0) 
LR_LIST=(1e-3 1e-4) 
AGG_LIST=(mean sum softmax softmin) 

# Global variables to track the best model across ALL hyperparameter combinations
GLOBAL_BEST_VAL_SCORE=10000
GLOBAL_BEST_MODEL_PATH=""
GLOBAL_BEST_PROCESSED_DIR=""
GLOBAL_BEST_PARAMS=""
GLOBAL_BEST_HIDDEN=""
GLOBAL_BEST_BITS=""
GLOBAL_BEST_AGG=""

# Keep track of all results for final summary
ALL_RESULTS=()

for BITS in "${BITS_LIST[@]}"; do
  for LEVEL in "${LEVEL_LIST[@]}"; do
    for RADIUS in "${RADIUS_LIST[@]}"; do
      # make seperate processed-data dir per (BITS,LEVEL,RADIUS)
      PROCESSED_DATA_PATH=$PWD/data/$DATA_NAME/$MODEL_NAME/processed/b${BITS}_l${LEVEL}_r${RADIUS}
      mkdir -p "$PROCESSED_DATA_PATH"
      echo "➤ Preprocessing: bits=$BITS, level=$LEVEL, radius=$RADIUS" | tee -a "$LOG_FILE"
      python $SCRIPT_BASE/preprocess.py \
        --raw_dir        "$RAW_DIR" \
        --pickle_dir     "$PICKLE_DIR" \
        --processed_dir  "$PROCESSED_DATA_PATH" \
        --num_workers    "$NUM_CORES" \
        --log_dir        "$PWD/logs/$MODEL_NAME/$DATA_NAME/prep_b${BITS}_l${LEVEL}_r${RADIUS}" \
        --bits           "$BITS" \
        --level          "$LEVEL" \
        --radius         "$RADIUS"
      
      for HIDDEN in "${HIDDEN_LIST[@]}"; do
        for LR in "${LR_LIST[@]}"; do
          for AGG in "${AGG_LIST[@]}"; do
            EXP_NAME="h${HIDDEN}_b${BITS}_l${LEVEL}_r${RADIUS}_lr${LR}_${AGG}"
            MODEL_PATH=$PWD/checkpoints/$MODEL_NAME/$DATA_NAME/$EXP_NAME
            LOG_PATH=$PWD/logs/$MODEL_NAME/$DATA_NAME/$EXP_NAME

            if [ -f "$MODEL_PATH/best_val_score.txt" ]; then
              echo "Skipping: Already trained - $EXP_NAME" | tee -a "$LOG_FILE"
            else
              echo "Training: hidden=$HIDDEN, lr=$LR, agg=$AGG" | tee -a "$LOG_FILE"
              python $SCRIPT_BASE/train.py \
                --processed_dir  "$PROCESSED_DATA_PATH" \
                --checkpoint_dir "$MODEL_PATH" \
                --log_dir        "$LOG_PATH" \
                --hidden         "$HIDDEN" \
                --batch_size     "$BATCH_SIZE" \
                --lr             "$LR" \
                --epochs         "$EPOCHS" \
                --patience       "$PATIENCE" \
                --agg            "$AGG" \
                --num_workers    "$NUM_CORES" \
                --device         "cuda:1" \
                --bits           "$BITS"
            fi

            
            # Check validation score from the log or saved metric file
            if [ -f "$MODEL_PATH/best_val_score.txt" ]; then
              VAL_SCORE=$(cat "$MODEL_PATH/best_val_score.txt")
              
              # Store this result for final summary
              RESULT="VAL_SCORE=$VAL_SCORE, hidden=$HIDDEN, bits=$BITS, level=$LEVEL, radius=$RADIUS, lr=$LR, agg=$AGG"
              ALL_RESULTS+=("$RESULT")
              
              # Check if this is the global best model so far
              if (( $(echo "$VAL_SCORE < $GLOBAL_BEST_VAL_SCORE" | bc -l) )); then
                GLOBAL_BEST_VAL_SCORE=$VAL_SCORE
                GLOBAL_BEST_MODEL_PATH=$MODEL_PATH
                GLOBAL_BEST_PROCESSED_DIR=$PROCESSED_DATA_PATH
                GLOBAL_BEST_PARAMS="hidden=$HIDDEN, bits=$BITS, level=$LEVEL, radius=$RADIUS, lr=$LR, agg=$AGG"
                GLOBAL_BEST_HIDDEN=$HIDDEN
                GLOBAL_BEST_BITS=$BITS
                GLOBAL_BEST_AGG=$AGG
                
                echo "===> New best model found! Val score: $GLOBAL_BEST_VAL_SCORE" | tee -a "$LOG_FILE"
              fi
            fi
          done
        done
      done
    done
  done
done

# # Sort results by validation score (highest first)
IFS=$'\n'
SORTED_RESULTS=($(sort -rn -t= -k2 <<<"${ALL_RESULTS[*]}"))
unset IFS

# Print summary of top 5 models
echo "================================================" | tee -a "$LOG_FILE"
echo "Top 5 Models by Validation Performance:" | tee -a "$LOG_FILE"
for i in {0..4}; do
  if [ $i -lt ${#SORTED_RESULTS[@]} ]; then
    echo "$((i+1)). ${SORTED_RESULTS[$i]}" | tee -a "$LOG_FILE"
  fi
done

# After evaluating all hyperparameter combinations, predict on test set using the GLOBAL best model
if [ -n "$GLOBAL_BEST_MODEL_PATH" ]; then
  echo "================================================" | tee -a "$LOG_FILE"
  echo "GLOBAL BEST MODEL" | tee -a "$LOG_FILE"
  echo "Parameters: $GLOBAL_BEST_PARAMS" | tee -a "$LOG_FILE"
  echo "Validation score: $GLOBAL_BEST_VAL_SCORE" | tee -a "$LOG_FILE"
  
  # Create output directory for final test predictions
  FINAL_TEST_OUTPUT_PATH=$PWD/results/$MODEL_NAME/$DATA_NAME/
  mkdir -p "$FINAL_TEST_OUTPUT_PATH"
  
  echo "Predicting on test set using global best model..."
  python $SCRIPT_BASE/03_predict.py \
    --processed_dir   "$GLOBAL_BEST_PROCESSED_DIR" \
    --checkpoint_dir  "$GLOBAL_BEST_MODEL_PATH" \
    --result_dir      "$FINAL_TEST_OUTPUT_PATH" \
    --log_dir         "$GLOBAL_BEST_MODEL_PATH/test_prediction" \
    --bits            "$GLOBAL_BEST_BITS" \
    --hidden          "$GLOBAL_BEST_HIDDEN" \
    --agg             "$GLOBAL_BEST_AGG" \
    --device          "cuda:1"
  
  echo "Test prediction complete. Results saved to: $FINAL_TEST_OUTPUT_PATH"
  
# Log final execution time
echo "================================================" | tee -a "$LOG_FILE"
echo "Script completed at: $(date)" | tee -a "$LOG_FILE"
echo "Total execution time: $SECONDS seconds" | tee -a "$LOG_FILE"
else
  echo "ERROR: No valid model found across all hyperparameter combinations!" | tee -a "$LOG_FILE"
  exit 1
fi