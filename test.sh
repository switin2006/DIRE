%%writefile test_cifake.sh
#!/bin/bash
EXP_NAME="cifake_classifier_eval"
# This path points to the 'best' model saved by the training script in the previous step.
CKPT_PATH="/content/DIRE/data/exp/cifake_classifier/ckpt/model_epoch_best.pth"
DATASETS_TEST="cifake"

# Ensure the symbolic link to the data folder exists
ln -sfn /content/data_DIRE /content/DIRE/data

# Run evaluation on a single GPU (gpus 0)
python /content/DIRE/test.py --gpus 0 --ckpt $CKPT_PATH --exp_name $EXP_NAME datasets_test $DATASETS_TEST
