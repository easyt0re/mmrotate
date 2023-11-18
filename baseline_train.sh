#!/bin/bash

# Set the path to the configuration file and checkpoint file
config_list=( "configs/oriented_rcnn/vanilla_o_rcnn_SRSDD.py" \
              "configs/oriented_reppoints/vanilla_o_rep_SRSDD.py" \
              "configs/rotated_faster_rcnn/vanilla_r_frcnn_SRSDD.py" \
              "configs/rotated_retinanet/vanilla_r_ret_SRSDD.py" \
              "configs/oriented_rcnn/vanilla_o_rcnn_CASIA.py" \
              "configs/oriented_reppoints/vanilla_o_rep_CASIA.py" \
              "configs/rotated_faster_rcnn/vanilla_r_frcnn_CASIA.py" \
              "configs/rotated_retinanet/vanilla_r_ret_CASIA.py" \
)

# Loop through the elements of the array
for i in "${!config_list[@]}"; do
  # Test model
  echo "Train model $i..."
  python tools/train.py ${config_list[$i]}
  # Check if model 1 was tested successfully
  if [ $? -eq 0 ]; then
    echo "Model $i trained successfully."
  else
    echo "Model $i failed to train."
    exit 1 # Exit the script with an error code
  fi
done

exit 0
