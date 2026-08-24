from __future__ import absolute_import, division, print_function

import argparse
import json
import os
from path_config import data_path


file_dir = os.path.dirname(__file__)


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="PRISM training and evaluation options")

        # Paths
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=data_path("c3vd"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))

        # Training
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="name of the folder to save the model in",
                                 default="prism")
        self.parser.add_argument("--edge_loss",
                                 help="enable edge-guided reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 default="c3vd_mysplit",
                                 choices=[
                                     "c3vd_mysplit",
                                     "c3vd_mysplit_interval5",
                                     "c3vd_mysplit_interval10",
                                     "c3vd_mysplit_interval20",
                                     "c3vd_mysplit_interval30",
                                     "hk",
                                     "hk_interval5",
                                     "hk_interval10",
                                     "hk_interval20",
                                     "hk_interval30",
                                 ])
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of ResNet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset loader to train on",
                                 default="hk",
                                 choices=["hk"])
        self.parser.add_argument("--png",
                                 help="train from png files instead of jpg files",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=288)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=288)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",
                                 help="use stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # Optimization
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=30)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)
        self.parser.add_argument("--supervised_alpha",
                                 type=float,
                                 help="supervised loss weight",
                                 default=0.1)
        self.parser.add_argument("--training_mode",
                                 type=str,
                                 help="which branch to train",
                                 default="both",
                                 choices=["pose", "depth", "both"])

        # Ablations
        self.parser.add_argument("--v1_multiscale",
                                 help="use monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="use average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="disable automasking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="use predictive masking",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="disable SSIM loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network receives",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="pose network type",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # System
        self.parser.add_argument("--no_cuda",
                                 help="disable CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # Loading
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="folder containing model weights")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # Logging
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between tensorboard logs",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between checkpoints",
                                 default=10)

        # Evaluation
        self.parser.add_argument("--eval_stereo",
                                 help="evaluate in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="evaluate in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="disable median scaling during evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="multiply predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="c3vd_mysplit",
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="save predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="disable evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="evaluate eigen predictions using benchmark files",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="folder for disparity outputs",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="use monodepth flip post-processing",
                                 action="store_true")

        # PRISM-specific data controls
        self.parser.add_argument("--method",
                                 type=str,
                                 help="backbone family for prediction scripts",
                                 choices=["monodepth", "monodepth2", "monovit", "IID"],
                                 default="monodepth")
        self.parser.add_argument("--input_mask_path",
                                 type=str,
                                 help="optional input mask path",
                                 default=None)
        self.parser.add_argument("--distorted",
                                 help="use distorted C3VD intrinsics",
                                 action="store_true")
        self.parser.add_argument("--config",
                                 type=str,
                                 help="JSON config file; values override CLI/defaults",
                                 default=None)
        self.parser.add_argument("--aug_type",
                                 type=str,
                                 help="dataset augmentation suffix",
                                 default="",
                                 choices=["", "add", "rem", "addrem"])
        self.parser.add_argument("--data",
                                 type=str,
                                 help="endoscopy dataset variant",
                                 default="c3vd",
                                 choices=["c3vd", "hk"])

    def parse(self):
        config_parser = argparse.ArgumentParser(add_help=False)
        config_parser.add_argument("--config", type=str)
        config_args, _ = config_parser.parse_known_args()
        if config_args.config is not None:
            with open(config_args.config, "r") as f:
                config = json.load(f)
            known = {action.dest for action in self.parser._actions}
            unknown = sorted(set(config) - known)
            if unknown:
                self.parser.error("unknown config keys: {}".format(", ".join(unknown)))
            self.parser.set_defaults(**config)
        self.options = self.parser.parse_args()
        return self.options
