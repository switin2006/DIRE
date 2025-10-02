"""
Modified from guided-diffusion/scripts/image_sample.py for single-GPU execution.
"""

import argparse
import os
import torch

import sys
import cv2
# Multi-GPU dependencies (MPI, torch.distributed) have been removed.

import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import torch as th

from guided_diffusion import dist_util, logger
# The original load_data_for_reverse is MPI-dependent, so we use a standard DataLoader setup.
from torch.utils.data import DataLoader
from guided_diffusion.image_datasets import ImageDataset_for_reverse
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def reshape_image(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    if imgs.shape[2] != imgs.shape[3]:
        crop_func = transforms.CenterCrop(image_size)
        imgs = crop_func(imgs)
    if imgs.shape[2] != image_size:
        imgs = F.interpolate(imgs, size=(image_size, image_size), mode="bicubic")
    return imgs


def main():
    args = create_argparser().parse_args()

    # setup_dist is removed as it's for multi-GPU.
    # The script will automatically use the GPU specified by CUDA_VISIBLE_DEVICES.
    logger.configure(dir=args.recons_dir)

    os.makedirs(args.recons_dir, exist_ok=True)
    os.makedirs(args.dire_dir, exist_ok=True)
    logger.log(str(args))

    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    
    # Use standard torch.load for single-GPU model loading.
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    
    model.to(dist_util.dev())
    logger.log("have created model and diffusion")
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # --- Simplified Data Loading for Single GPU ---
    # Manually create the dataset and dataloader without MPI dependencies.
    from guided_diffusion.image_datasets import _list_image_files_recursively
    all_files = _list_image_files_recursively(args.images_dir)
    
    # --- THIS SECTION IS CORRECTED ---
    # The 'class_cond' argument is removed as it's not a valid parameter for the class constructor.
    # The class correctly handles cases where classes are None.
    dataset = ImageDataset_for_reverse(
        resolution=args.image_size,
        image_paths=all_files,
        shard=0, # shard is always 0 for single GPU
        num_shards=1, # num_shards is always 1 for single GPU
    )
    data = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    logger.log("have created data loader")
    # --- End of Correction ---

    logger.log("computing recons & DIRE ...")
    have_finished_images = 0
    
    # Use a simple 'for' loop for single-GPU processing.
    for imgs, out_dicts, paths in data:
        if have_finished_images >= args.num_samples and args.num_samples != -1:
            break

        batch_size = imgs.shape[0]

        imgs = imgs.to(dist_util.dev())
        model_kwargs = {}
        if args.class_cond:
            # Note: This creates random classes. If you have true labels, they should be loaded here.
            classes = th.randint(low=0, high=NUM_CLASSES, size=(batch_size,), device=dist_util.dev())
            model_kwargs["y"] = classes

        reverse_fn = diffusion.ddim_reverse_sample_loop
        imgs = reshape_image(imgs, args.image_size)

        latent = reverse_fn(
            model,
            (batch_size, 3, args.image_size, args.image_size),
            noise=imgs,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            real_step=args.real_step,
        )
        sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        recons = sample_fn(
            model,
            (batch_size, 3, args.image_size, args.image_size),
            noise=latent,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            real_step=args.real_step,
        )

        dire = th.abs(imgs - recons)
        recons = ((recons + 1) * 127.5).clamp(0, 255).to(th.uint8)
        recons = recons.permute(0, 2, 3, 1)
        recons = recons.contiguous()

        imgs = ((imgs + 1) * 127.5).clamp(0, 255).to(th.uint8)
        imgs = imgs.permute(0, 2, 3, 1)
        imgs = imgs.contiguous()

        dire = (dire * 255.0 / 2.0).clamp(0, 255).to(th.uint8)
        dire = dire.permute(0, 2, 3, 1)
        dire = dire.contiguous()

        # No gathering needed for single GPU. Process tensors directly.
        recons_np = recons.cpu().numpy()
        dire_np = dire.cpu().numpy()

        for i in range(len(recons_np)):
            if args.has_subfolder:
                # Assumes paths are structured like ".../class_name/image.png"
                subfolder = os.path.basename(os.path.dirname(paths[i]))
                recons_save_dir = os.path.join(args.recons_dir, subfolder)
                dire_save_dir = os.path.join(args.dire_dir, subfolder)
            else:
                recons_save_dir = args.recons_dir
                dire_save_dir = args.dire_dir

            fn_save = os.path.basename(paths[i])
            os.makedirs(recons_save_dir, exist_ok=True)
            os.makedirs(dire_save_dir, exist_ok=True)

            cv2.imwrite(
                f"{dire_save_dir}/{fn_save}", cv2.cvtColor(dire_np[i].astype(np.uint8), cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(f"{recons_save_dir}/{fn_save}", cv2.cvtColor(recons_np[i].astype(np.uint8), cv2.COLOR_RGB2BGR))

        have_finished_images += batch_size
        logger.log(f"have finished {have_finished_images} samples")

    # dist.barrier() is removed.
    logger.log("finish computing recons & DIRE!")


def create_argparser():
    defaults = dict(
        images_dir="", # Generic default path
        recons_dir="", # Generic default path
        dire_dir="",   # Generic default path
        clip_denoised=True,
        num_samples=-1,
        batch_size=16,
        use_ddim=False,
        model_path="",
        real_step=0,
        continue_reverse=False,
        has_subfolder=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
