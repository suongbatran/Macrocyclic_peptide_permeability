#!/bin/bash

# Set environment variables
MODEL_NAME="Graph_3d_schnet"
SCRIPT_BASE="scripts/model/$MODEL_NAME"
PY_CMD="$SCRIPT_BASE"

# Change this tp "split_monomer" is you want to use other dataset
export DATA_NAME="split_random"

export TRAIN_FILE=$PWD/data/$DATA_NAME/raw_train.csv
export VAL_FILE=$PWD/data/$DATA_NAME/raw_val.csv
export TEST_FILE=$PWD/data/$DATA_NAME/raw_test.csv
export NUM_CORES=64

export PROCESSED_DATA_PATH=$PWD/data/$DATA_NAME/processed
export MODEL_PATH=$PWD/checkpoints/$MODEL_NAME/$DATA_NAME
export TEST_OUTPUT_PATH=$PWD/results/$MODEL_NAME/$DATA_NAME
export LOG_PATH=$PWD/logs/$MODEL_NAME/$DATA_NAME

# Check if the required files exist
[ -f $TRAIN_FILE ] || { echo "$TRAIN_FILE does not exist"; exit 1; }
[ -f $VAL_FILE ] || { echo "$VAL_FILE does not exist"; exit 1; }
[ -f $TEST_FILE ] || { echo "$TEST_FILE does not exist"; exit 1; }

# Hyperparameters
EPOCHS=250                      # Lightning max_epochs
PATIENCE=25                     # early-stop patience
GROUPS_PER_BATCH=12 

# Comma-lists below trigger grid-search; leave a single value for one run
HIDDEN="32,64,128"                 # hidden dims to sweep
LAYERS="3,4,5"                    # GNN layers to sweep
LR="1e-3,1e-4"                  # learning rates to sweep
AGG=""mean","softmax","softmin""    # pooling of conformer layer methods: "sum","mean","softmax","softmin"  
CUTOFF="10,8,6"

# Step 1: Preprocess the data
python $PY_CMD/preprocess.py \
    --raw_dir $PWD/data/$DATA_NAME \
    --pickle_dir $PWD/pickle \
    --processed_dir $PROCESSED_DATA_PATH \
    --num_workers $NUM_CORES \
    --log_dir $LOG_PATH \

# Step 2: Train the model (grid-search externally)
# parse comma‐lists into arrays
IFS=',' read -r -a HIDDEN_LIST <<< "$HIDDEN"
IFS=',' read -r -a LAYERS_LIST <<< "$LAYERS"
IFS=',' read -r -a LR_LIST     <<< "$LR"
# Note: remove the extra quotes around AGG
IFS=',' read -r -a AGG_LIST    <<< "$(echo $AGG | tr -d '"')"
IFS=',' read -r -a CUTOFF_LIST <<< "$CUTOFF"

# initialize global best
best_mse=100000000
$BEST_FILE=""
# ensure root best_gnn placeholder
mkdir -p "$MODEL_PATH"
touch  "$MODEL_PATH/best_gnn.pt"

export OMP_NUM_THREADS=$(( NUM_CORES / SLURM_GPUS_ON_NODE ))
for hidden in "${HIDDEN_LIST[@]}"; do
  for layers in "${LAYERS_LIST[@]}"; do
    for lr in "${LR_LIST[@]}"; do
      for agg in "${AGG_LIST[@]}"; do
        for cutoff in "${CUTOFF_LIST[@]}"; do

          # build a run‐name that matches your Python gridsearch
          run_name="hidden${hidden}_num_interactions${layers}_lr${lr}_agg${agg}_cutoff${cutoff}"
          combo_ckpt="$MODEL_PATH/$run_name"
          combo_log="$LOG_PATH/$run_name"
          mkdir -p "$combo_ckpt" "$combo_log"

          echo
          echo ">>> Starting combo: $run_name"
          echo

          # 1) Launch a fresh 4‐GPU DDP run for this combo
          torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE \
             "scripts/model/$MODEL_NAME/train.py" \
              --processed_dir    "$PROCESSED_DATA_PATH" \
              --checkpoint_dir   "$combo_ckpt" \
              --log_dir          "$combo_log" \
              --hidden           "$hidden" \
              --num_interactions "$layers" \
              --lr               "$lr" \
              --agg              "$agg" \
              --cutoff           "$cutoff" \
              --groups_per_batch "$GROUPS_PER_BATCH" \
              --epochs           "$EPOCHS" \
              --patience         "$PATIENCE" \
              --num_workers      "$NUM_CORES" \
              --devices          $SLURM_GPUS_ON_NODE

          # 2) Find this combo’s best checkpoint (lowest val_mse in filename)
            for f in "$combo_ckpt"/epoch=*-val_mse=*.ckpt; do
                # skip the literal glob if no files match
                [[ -e "$f" ]] || continue

                # pull out the float between "val_mse=" and ".ckpt"
                val=${f##*val_mse=}      # yields "1.5813.ckpt"
                val=${val%.ckpt}         # yields "1.5813"

                # compare, update if lower
                if awk "BEGIN{exit !($val < $best_mse)}"; then
                    best_mse=$val
                    BEST_FILE=$f
                fi
            done

            if [[ -n "$BEST_FILE" ]]; then
                cp "$BEST_FILE" "$MODEL_PATH/best_gnn.pt"
                echo
                echo "◆ GLOBAL BEST: valMSE=$best_mse → $MODEL_PATH/best_gnn.pt"
                echo "◆ BEST CONFIG: $BEST_FILE"
                echo
            else
                echo "⚠️  No checkpoint files found in $combo_ckpt"
            fi
        done
      done
    done
  done
done

echo
echo "+++ GRID‐SEARCH COMPLETE +++"
echo "◆ FINAL BEST valMSE=$best_mse"
echo "◆ FINAL BEST CONFIG: $best_cfg"
echo "◆ FINAL BEST CHECKPOINT: $MODEL_PATH/best_gnn.pt"

# Step 3: Make predictions
python $PY_CMD/03_predict.py \
    --processed_dir $PROCESSED_DATA_PATH \
    --checkpoint_dir $MODEL_PATH \
    --result_dir $TEST_OUTPUT_PATH \
    --log_dir $LOG_PATH \
    --agg "$AGG"
