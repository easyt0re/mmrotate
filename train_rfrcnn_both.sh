#!/bin/bash
# Train model 1
echo "Training model 1..."
python tools/train.py configs/rotated_retinanet/t_custom_r_ret_SRSDD.py
# Check if model 1 was trained successfully
if [ $? -eq 0 ]; then
  echo "Model 1 trained successfully."
else
  echo "Model 1 failed to train."
  exit 1 # Exit the script with an error code
fi

# Train model 2
echo "Training model 2..."
python tools/train.py configs/rotated_retinanet/t_custom_r_ret_CASIA.py
# Check if model 2 was trained successfully
if [ $? -eq 0 ]; then
  echo "Model 2 trained successfully."
else
  echo "Model 2 failed to train."
  exit 2 # Exit the script with a different error code
fi

# Train model 3
echo "Training model 3..."
python tools/train.py configs/oriented_reppoints/t_custom_o_rep_CASIA.py
# Check if model 3 was trained successfully
if [ $? -eq 0 ]; then
  echo "Model 3 trained successfully."
else
  echo "Model 3 failed to train."
  exit 3 # Exit the script with a different error code
fi

# If both models were trained successfully, exit with a success code
echo "Three models trained successfully."
exit 0
