#!/bin/bash

# Set the path to the configuration file and checkpoint file
CONFIG_FILE="configs/oriented_rcnn/custom_o_rcnn_SRSDD.py"
CHECKPOINT_FILE="ckpt/11.pth"
PKL_FILE="testpkls/11.pkl"

MAT_PATH="confMat/11/" # need to create this dir first, maybe one level is fine since need to save again (rename)

# Set the evaluation metrics
EVAL_METRICS="mAP"

# Test model 1
echo "Testing model 1..."
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
    --eval ${EVAL_METRICS} \
    --out ${PKL_FILE}
# Check if model 1 was tested successfully
if [ $? -eq 0 ]; then
  echo "Model 1 tested successfully."
else
  echo "Model 1 failed to test."
  exit 1 # Exit the script with an error code
fi
# generate confusion matrix
echo "generating confusion matrix for model 1..."
python tools/analysis_tools/confusion_matrix.py ${CONFIG_FILE} ${PKL_FILE} ${MAT_PATH} --show
# Check if model 1 was tested successfully
if [ $? -eq 0 ]; then
  echo "Model 1 gen mat successfully."
else
  echo "Model 1 failed to gen."
  exit 2 # Exit the script with an error code
fi
# If both models were tested successfully, exit with a success code
echo "done with model 1."
exit 0
