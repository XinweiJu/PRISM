# train_xinwei_edge2.py

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train_scripts.trainer_edge import Trainer
from options import MonodepthOptions
from path_config import weights_path

if __name__ == "__main__":
    options = MonodepthOptions()
    opts = options.parse()

    model_name = opts.model_name.replace("_edge_ssim_depth_fix_thorough_aug", "")
    model_base = weights_path("monodepth", model_name, "models", "weights_19")

    opts.model_load_paths = {
        "encoder": f"{model_base}/encoder.pth",
        "depth": f"{model_base}/depth.pth",
        "pose_encoder": f"{model_base}/pose_encoder.pth",
        "pose": f"{model_base}/pose.pth",
        # "pose_encoder": weights_path("checkpoints", "monodepth", "mono_640x192", "pose_encoder.pth"),
        # "pose": weights_path("checkpoints", "monodepth", "mono_640x192", "pose.pth"),
    }

    trainer = Trainer(opts)
    trainer.train()

# CUDA_VISIBLE_DEVICES=3 nohup python train_xinwei_edge_pose_from_mod.py --config configs_loss/monodepth_mono_hkfull_finetuned_edge_loss_ssim.json > logs/monodepth_mono_hkfull_finetuned_edge_loss_ssim.log 2>&1 &
