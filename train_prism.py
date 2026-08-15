from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

from options import MonodepthOptions
from path_config import weights_path


PIPELINES = {
    "joint": "standard joint depth/pose training with trainer.py",
    "edge": "stage-3 edge-guided pose fine-tuning with frozen depth",
    "depth": "legacy stage-3 pose fine-tuning with frozen depth",
    "edge_pose_scratch": "edge-guided training with pose initialized from a generic checkpoint",
    "edge_pose_resume": "edge-guided training with depth and pose initialized from a PRISM checkpoint",
}


def parse_pipeline(argv):
    parser = argparse.ArgumentParser(
        description="Unified PRISM training entrypoint",
        add_help=False,
    )
    parser.add_argument(
        "--pipeline",
        choices=sorted(PIPELINES),
        default="joint",
        help="training pipeline to run; remaining args are passed to options.py",
    )
    args, remaining = parser.parse_known_args(argv)
    return args.pipeline, remaining


def parse_training_options(remaining):
    old_argv = sys.argv
    sys.argv = [old_argv[0]] + remaining
    try:
        return MonodepthOptions().parse()
    finally:
        sys.argv = old_argv


def checkpoint_model_name(opts, suffix):
    if suffix and opts.model_name.endswith(suffix):
        return opts.model_name[: -len(suffix)]
    return opts.model_name


def configure_preload(opts, pipeline):
    if pipeline == "edge":
        opts.edge_loss = True
        return

    if pipeline == "edge_pose_scratch":
        opts.edge_loss = True
        model_name = checkpoint_model_name(opts, "_edge_ssim")
        model_base = weights_path("monodepth", model_name, "models", "weights_19")
        pose_base = weights_path("checkpoints", "monodepth", "mono_640x192")
        opts.model_load_paths = {
            "encoder": os.path.join(model_base, "encoder.pth"),
            "depth": os.path.join(model_base, "depth.pth"),
            "pose_encoder": os.path.join(pose_base, "pose_encoder.pth"),
            "pose": os.path.join(pose_base, "pose.pth"),
        }
        return

    if pipeline == "edge_pose_resume":
        opts.edge_loss = True
        model_name = checkpoint_model_name(opts, "_edge_ssim_depth_fix_thorough_aug")
        model_base = weights_path("monodepth", model_name, "models", "weights_19")
        opts.model_load_paths = {
            "encoder": os.path.join(model_base, "encoder.pth"),
            "depth": os.path.join(model_base, "depth.pth"),
            "pose_encoder": os.path.join(model_base, "pose_encoder.pth"),
            "pose": os.path.join(model_base, "pose.pth"),
        }


def trainer_for_pipeline(pipeline):
    if pipeline == "depth":
        from ablations.training.trainer_depth import Trainer
    elif pipeline in ["edge", "edge_pose_scratch", "edge_pose_resume"]:
        from ablations.training.trainer_edge import Trainer
    else:
        from trainer import Trainer
    return Trainer


def main():
    pipeline, remaining = parse_pipeline(sys.argv[1:])
    opts = parse_training_options(remaining)
    configure_preload(opts, pipeline)
    Trainer = trainer_for_pipeline(pipeline)

    print("Running PRISM pipeline: {} ({})".format(pipeline, PIPELINES[pipeline]))
    trainer = Trainer(opts)
    trainer.train()


if __name__ == "__main__":
    main()
