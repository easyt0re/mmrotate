#!/bin/bash

# Set the path to the configuration file and checkpoint file
config_list=( "configs/oriented_rcnn/custom_o_rcnn_SRSDD.py" \
              "configs/oriented_reppoints/custom_o_rep_SRSDD.py" \
              "configs/rotated_faster_rcnn/custom_r_frcnn_SRSDD.py" \
              "configs/rotated_retinanet/custom_r_ret_SRSDD.py" \
              "configs/oriented_rcnn/custom_o_rcnn_CASIA.py" \
              "configs/oriented_reppoints/custom_o_rep_CASIA.py" \
              "configs/rotated_faster_rcnn/custom_r_frcnn_CASIA.py" \
              "configs/rotated_retinanet/custom_r_ret_CASIA.py" \
)

ckpt_list=( "ckpt/11.pth" \
            "ckpt/21.pth" \
            "ckpt/31.pth" \
            "ckpt/41.pth" \
            "ckpt/12.pth" \
            "ckpt/22.pth" \
            "ckpt/32.pth" \
            "ckpt/42.pth" \
)

pkl_list=("testpkls/11.pkl" \
          "testpkls/21.pkl" \
          "testpkls/31.pkl" \
          "testpkls/41.pkl" \
          "testpkls/12.pkl" \
          "testpkls/22.pkl" \
          "testpkls/32.pkl" \
          "testpkls/42.pkl" \
)

# need to create this dir first
mat_list=("confMat/11/" \
          "confMat/21/" \
          "confMat/31/" \
          "confMat/41/" \
          "confMat/12/" \
          "confMat/22/" \
          "confMat/32/" \
          "confMat/42/" \
)

# Set the evaluation metrics
EVAL_METRICS="mAP"

# Loop through the elements of the array
for i in "${!config_list[@]}"; do
  # Test model
  echo "Testing model $i..."
  python tools/test.py ${config_list[$i]} ${ckpt_list[$i]} \
      --eval ${EVAL_METRICS} \
      --out ${pkl_list[$i]}
  # Check if model 1 was tested successfully
  if [ $? -eq 0 ]; then
    echo "Model $i tested successfully."
  else
    echo "Model $i failed to test."
    exit 1 # Exit the script with an error code
  fi
  # generate confusion matrix
  echo "generating confusion matrix for model $i..."
  python tools/analysis_tools/confusion_matrix.py \
    ${config_list[$i]} ${pkl_list[$i]} ${mat_list[$i]} --show
  # Check if model 1 was tested successfully
  if [ $? -eq 0 ]; then
    echo "Model $i gen mat successfully."
  else
    echo "Model $i failed to gen."
    exit 2 # Exit the script with an error code
  fi
  # If both models were tested successfully, exit with a success code
  echo "done with model $i."
done

exit 0
