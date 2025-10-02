## set MODEL_PATH, num_samples, has_subfolder, images_dir, recons_dir, dire_dir
%%writefile run_compute_dire.sh
#!/bin/bash
MODEL_PATH="/content/cifar10_uncond_50M_500K.pt"
CIFAKE_DATASET_ROOT="/content/cifake_subset"
OUTPUT_ROOT="/content/generated_DIRE"
export CUDA_VISIBLE_DEVICES=0

MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 --class_cond False --attention_resolutions 16,8 --diffusion_steps 4000 --noise_schedule cosine"
SAMPLE_FLAGS="--batch_size 64 --num_samples -1 --timestep_respacing ddim20 --use_ddim True"

for split in train test; do
    for type in REAL FAKE; do
        IMAGES_DIR="$CIFAKE_DATASET_ROOT/$split/$type"
        RECONS_DIR="$OUTPUT_ROOT/$split/$type/recons"
        DIRE_DIR="$OUTPUT_ROOT/$split/$type/dire"

        mkdir -p "$RECONS_DIR" "$DIRE_DIR"

        SAVE_FLAGS="--images_dir $IMAGES_DIR --recons_dir $RECONS_DIR --dire_dir $DIRE_DIR --has_subfolder False" # has_subfolder is False here

        echo "--- PROCESSING: $IMAGES_DIR ---"
        python /content/DIRE/guided-diffusion/compute_dire.py \
            --model_path "$MODEL_PATH" $MODEL_FLAGS $SAMPLE_FLAGS $SAVE_FLAGS
    done
done
echo "✅ STAGE 1 COMPLETE: DIRE computation finished."
