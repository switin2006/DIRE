%%writefile train_cifake.sh
#!/bin/bash
EXP_NAME="cifake_classifier"
DATASETS="cifake"
DATASETS_TEST="cifake" # This is used for validation during training

# Create a symbolic link so the script can find the data at the expected 'data' path
# The train.py script looks for a folder named 'data' in the root project directory.
ln -sfn /content/data_DIRE /content/DIRE/data

# Run training on a single GPU (gpus 0)
python /content/DIRE/train.py --gpus 0 --exp_name $EXP_NAME datasets $DATASETS datasets_test $DATASETS_TEST
