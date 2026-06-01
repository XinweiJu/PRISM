# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time
from itertools import chain
from PIL import Image

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed

import glob
from sklearn.model_selection import train_test_split
import random 
from torchvision.transforms import transforms

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        # self.parameters_to_train = []
        self.depth_parameters = []
        self.pose_parameters = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.training_mode in ["depth", "both"]:
            self.channels_per_image_depth = 4 
        else:
            self.channels_per_image_depth = 3

        if self.opt.training_mode in ["pose", "both"]:
            self.channels_per_image_pose = 4  
        else:
            self.channels_per_image_pose = 3

        self.depth_use_lum = False
        self.depth_use_edge = False
        self.pose_use_edge = False
        self.pose_use_lum = False

        if "dlpel" in self.opt.model_name:
            self.depth_use_lum = True
            self.pose_use_edge = True
            self.pose_use_lum = True
        if "dlpe" in self.opt.model_name:
            self.depth_use_lum = True
            self.pose_use_edge = True
        elif "depl" in self.opt.model_name:
            self.depth_use_edge = True
            self.pose_use_lum = True
        elif "both_edge" in self.opt.model_name:
            self.depth_use_edge = True
            self.pose_use_edge = True
        elif "both_lum" in self.opt.model_name:
            self.depth_use_lum = True
            self.pose_use_lum = True
        elif "depth_lum" in self.opt.model_name:
            self.depth_use_lum = True
        elif "depth_edge" in self.opt.model_name:
            self.depth_use_edge = True
        elif "pose_edge" in self.opt.model_name:
            self.pose_use_edge = True
        elif "pose_lum" in self.opt.model_name:
            self.pose_use_lum = True


        if any(key in self.opt.model_name for key in ["edge_dice", "edge_ssim"]):
                self.opt.edge_loss = True

        for key in ["edge_dice", "edge_ssim"]:
            if key in self.opt.model_name:
                self.edge_loss_type = key
                break
        else:
            self.edge_loss_type = None



        print("self.channels_per_image_depth", self.channels_per_image_depth)
        print("self.channels_per_image_pose", self.channels_per_image_pose)
        print("self.depth_use_lum", self.depth_use_lum)
        print("self.depth_use_edge", self.depth_use_edge)
        print("self.pose_use_edge", self.pose_use_edge)
        print("self.pose_use_lum", self.pose_use_lum)
        print("self.opt.edge_loss", self.opt.edge_loss)
        print("self.edge_loss_type", self.edge_loss_type)


        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        import networks

        if options.method == "monodepth2":
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=1,
                channels_per_image=self.channels_per_image_depth
            )
            
            self.models["encoder"].to(self.device)
            # self.depth_parameters += list(self.models["encoder"].parameters())
            self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        elif options.method == "monovit":
            import networksMonoVIT as networksvit
            self.models["encoder"] = networksvit.mpvit_small()
            self.models["encoder"].to(self.device)
            self.models["depth"] = networksvit.DepthDecoder()
            self.depth_parameters += list(self.models["encoder"].parameters())
            
        self.models["depth"].to(self.device)
        # self.depth_parameters += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames,
                    channels_per_image=self.channels_per_image_pose
                )

                self.models["pose_encoder"].to(self.device)
                self.pose_parameters += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.pose_parameters += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.depth_parameters += list(self.models["predictive_mask"].parameters())

        if options.method == "monodepth2":
            # self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)

            # self.depth_optimizer = optim.Adam(self.depth_parameters, self.opt.learning_rate)
            self.pose_optimizer = optim.Adam(self.pose_parameters, self.opt.learning_rate)

            # self.depth_lr_scheduler = optim.lr_scheduler.StepLR(
            #     self.depth_optimizer, self.opt.scheduler_step_size, 0.1)
            self.pose_lr_scheduler = optim.lr_scheduler.StepLR(
                self.pose_optimizer, self.opt.scheduler_step_size, 0.1)
            
        elif options.method == "monovit":
            self.params = [{
                "params": self.parameters_to_train,
                "lr": 1e-4
                # "weight_decay": 0.01
            },
                {
                    "params": list(self.models["encoder"].parameters()),
                    "lr": self.opt.learning_rate
                    # "weight_decay": 0.01
                }]

            self.model_optimizer = optim.AdamW(self.params)

            # self.model_optimizer = optim.AdamW(self.parameters_to_train,1e-4,weight_decay=0.01)

            # self.model_optimizer = optim.AdamW(list(self.models["encoder"].parameters()), self.opt.learning_rate,
            #                                    weight_decay=0.01)

            self.model_lr_scheduler = optim.lr_scheduler.ExponentialLR(
                self.model_optimizer, 0.9)

        # if self.opt.load_weights_folder is not None:
        #     # Step 1: load depth-related weights
        #     self.opt.models_to_load = ["encoder", "depth"]
        #     self.opt.load_weights_folder = "/Datasets/Weights_Feast_all/monodepth/hk_mono_finetuned_both_lum/models/weights_19"
        #     self.load_model()

        #     # Step 2: load pose-related weights
        #     self.opt.models_to_load = ["pose_encoder", "pose"]
        #     self.opt.load_weights_folder = "/Datasets/Checkpoints/monodepth/mono_640x192"
        #     self.load_model()

        # self.opt.model_load_paths = {
        #     "encoder": "/Datasets/Weights_Feast_all/monodepth/hk_mono_finetuned_dlpe/models/weights_19/encoder.pth",
        #     "depth": "/Datasets/Weights_Feast_all/monodepth/hk_mono_finetuned_dlpe/models/weights_19/depth.pth",
        #     "pose_encoder": "/Datasets/Checkpoints/monodepth/mono_640x192/pose_encoder.pth",
        #     "pose": "/Datasets/Checkpoints/monodepth/mono_640x192/pose.pth",
        # }
        self.opt.models_to_load = ["encoder", "depth", "pose_encoder", "pose"]
        self.load_model()
        self.freeze_depth_networks()


        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)
        print("Training split with:\n  ", self.opt.split)
        print("Training with frame ids:\n  ", self.opt.frame_ids)
        print("Disable automasking is set to:\n  ", self.opt.disable_automasking)   
        print("Number of workers:\n  ", self.opt.num_workers)
        print("training_mode:\n  ", self.opt.training_mode)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "hk": datasets.HKDataset,
                         }
        self.dataset = datasets_dict[self.opt.dataset]

        if not isinstance(self.opt.split, list):
            self.opt.split = [self.opt.split]
            
        fpath = [os.path.join(os.path.dirname(__file__), "splits", split, "{}_files.txt") for split in self.opt.split]        
        img_ext = '.png' if self.opt.png else '.jpg'
        # it is ok if the list is random, in mono_dataset this will be dealt with using folder and frame_index
        # but i sort them eitherway for the validation and training list to be the same frames from each augmentation
        val_filenames =[]
        train_filenames = []
        for i, split in enumerate(self.opt.split):
            data_path = self.opt.data_path[i]
            if split == "hk" or split == "c3vd":
                aug = self.opt.aug_type
                if not os.path.exists(fpath[i].format(f'train{aug}')):
                    sub_files =[]
                    if not isinstance(data_path, list):
                        data_path = [data_path]
                    
                    traintestval = []
                    for data_pathi in data_path:
                        contents_lists = glob.glob(os.path.join(data_pathi, "*"))
                        sub_files =[]
                        for subdir in contents_lists:
                            sub_files.append(sorted(glob.glob(os.path.join(subdir,f"*{img_ext}")))[1:-1])
                        all_files = list(chain.from_iterable(sub_files))
                        train_temp_f, test_f = train_test_split(all_files, test_size=0.04, shuffle=False)
                        train_f, val_f = train_test_split(train_temp_f, test_size=0.1, shuffle=False)
                        traintestval.append([train_f, test_f, val_f])
                    train_filenames = list(chain.from_iterable([traintestval[i][0] for i in range(len(traintestval))]))
                    test_filenames = list(chain.from_iterable([traintestval[i][1] for i in range(len(traintestval))]))
                    val_filenames = list(chain.from_iterable([traintestval[i][2] for i in range(len(traintestval))]))
                    
                    # Extract the directory from the file path pattern
                    directory = os.path.dirname(fpath[i])

                    # Ensure the directory exists
                    os.makedirs(directory, exist_ok=True)
                    
                    # Save train_filenames to a text file
                    with open(fpath[i].format(f"train{aug}"), 'w') as f:
                        for filename in train_filenames:
                            f.write("%s\n" % filename)

                    # Save test_filenames to a text file
                    with open(fpath[i].format(f"test{aug}"), 'w') as f:
                        for filename in test_filenames:
                            f.write("%s\n" % filename)

                    # Save val_filenames to a text file
                    with open(fpath[i].format(f"val{aug}"), 'w') as f:
                        for filename in val_filenames:
                            f.write("%s\n" % filename)
                else:
                    train_filenames.extend(readlines(fpath[i].format(f"train{aug}")))
                    val_filenames.extend(readlines(fpath[i].format(f"val{aug}")))
                
            else:
                train_filenames.extend(readlines(fpath[i].format("train")))
                val_filenames.extend(readlines(fpath[i].format("val")))
        
        
        self.input_mask = {}
        for scale in self.opt.scales:
            if self.opt.input_mask_path is not None:
                input_mask_np = np.array(Image.open(self.opt.input_mask_path).resize((self.opt.width // (2 ** scale), self.opt.height // (2 ** scale)), Image.BOX))
                self.input_mask[scale] = torch.from_numpy(input_mask_np).to(self.device).unsqueeze(dim=0).repeat(12,1,1)
            else:
                self.input_mask[scale] = None
            
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        print("self.opt.data:", self.opt.data)
        if self.opt.dataset == "hk":
            train_dataset = self.dataset(
                self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, data = self.opt.data, distorted = self.opt.distorted)
        else:
            train_dataset = self.dataset(
                self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        
        # generator = torch.Generator()
        # generator.manual_seed(42)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        if self.opt.dataset == "hk":
            val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, data = self.opt.data, distorted = self.opt.distorted)
        else:
            val_dataset = self.dataset(
                self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
            
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def freeze_depth_networks(self):
        print("🔒 Freezing depth networks...")
        
        # 冻结encoder
        if "encoder" in self.models:
            for param in self.models["encoder"].parameters():
                param.requires_grad = False
            print("  ✅ Encoder frozen")
        
        # 冻结depth decoder
        if "depth" in self.models:
            for param in self.models["depth"].parameters():
                param.requires_grad = False
            print("  ✅ Depth decoder frozen")
        
        # 如果有predictive_mask也冻结
        if "predictive_mask" in self.models:
            for param in self.models["predictive_mask"].parameters():
                param.requires_grad = False
            print("  ✅ Predictive mask frozen")

    def set_train(self):
        """Convert all models to training mode
        """
        if "pose_encoder" in self.models:
            self.models["pose_encoder"].train()
        if "pose" in self.models:
            self.models["pose"].train()
        
        if "encoder" in self.models:
            self.models["encoder"].eval()
        if "depth" in self.models:
            self.models["depth"].eval()
        if "predictive_mask" in self.models:
            self.models["predictive_mask"].eval()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            # ---- Step 1: backward for depth
            # self.depth_optimizer.zero_grad()
            # losses["loss_depth"].backward(retain_graph=True)

            # ---- Step 2: backward for pose
            if self.use_pose_net:
                self.pose_optimizer.zero_grad()
                losses["loss_pose"].backward()

            # ---- Check gradients BEFORE step
            # print("[Gradient Norm Check]")
            # for name, param in self.models["pose_encoder"].named_parameters():
            #     if param.grad is not None:
            #         print(f"[POSE] {name} grad norm: {param.grad.norm():.4f}")

            # for name, param in self.models["encoder"].named_parameters():
            #     if param.grad is not None:
            #         print(f"[DEPTH] {name} grad norm: {param.grad.norm():.4f}")

            # ---- Step 3: optimizer step
            # self.depth_optimizer.step()
                self.pose_optimizer.step()


            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss_pose"].cpu().data)  # 🔥 修改：显示pose loss

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        # self.depth_lr_scheduler.step()
        self.pose_lr_scheduler.step()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            with torch.no_grad():
                if self.opt.training_mode in ["depth", "both"]:
                    if self.depth_use_lum == True:
                        inputs_feature = torch.cat([inputs[("color_aug", 0, 0)], inputs[("lum", 0, 0)]], dim=1)
                    elif self.depth_use_edge == True:
                        inputs_feature = torch.cat([inputs[("color_aug", 0, 0)], inputs[("edge", 0, 0)]], dim=1)
                else:
                    inputs_feature = inputs[("color_aug", 0, 0)]
                features = self.models["encoder"](inputs_feature)
                outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            with torch.no_grad():  
                outputs["predictive_mask"] = self.models["predictive_mask"](features)


        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)

        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                if self.opt.training_mode in ["pose", "both"]:
                    if self.pose_use_edge == True:
                        pose_feats = {
                            f_i: torch.cat([inputs[("color_aug", f_i, 0)], inputs[("edge", f_i, 0)]], dim=1)
                            for f_i in self.opt.frame_ids
                        }
                    elif self.pose_use_lum == True:
                        pose_feats = {
                                f_i: torch.cat([inputs[("color_aug", f_i, 0)], inputs[("lum", f_i, 0)]], dim=1)
                                for f_i in self.opt.frame_ids
                            }
                else:
                    pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            with torch.no_grad():
                disp = outputs[("disp", scale)]
                if self.opt.v1_multiscale:
                    source_scale = scale
                else:
                    disp = F.interpolate(
                        disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    source_scale = 0

                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                
            outputs[("depth", 0, scale)] = depth.detach()

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth.detach(), inputs[("inv_K", source_scale)])  
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)  
                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")
                
                outputs[("edge", frame_id, scale)] = F.grid_sample(
                    inputs[("edge", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_dice_loss(self, pred, target, eps=1e-6):
        """Computes Dice loss between two edge maps"""
        pred = torch.sigmoid(pred) if pred.dtype == torch.float32 else pred
        target = target.float()

        intersection = (pred * target).sum(dim=(1,2,3))
        union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))

        dice = (2. * intersection + eps) / (union + eps)
        return 1 - dice.unsqueeze(1).unsqueeze(1)  # keep dims like reprojection loss


    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection, edge, and smoothness losses for a minibatch"""
        losses = {}
        total_loss = 0
        if self.opt.edge_loss:
            total_loss_edge = 0

        edge_loss_sum_all = 0
        edge_loss_count_all = 0

        for scale in self.opt.scales:
            loss = 0
            if self.opt.edge_loss:
                edge_loss = 0
            reprojection_losses = []
            reprojection_losses_edge = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
            target_edge = inputs[("edge", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_loss = self.compute_reprojection_loss(pred, target)
                reprojection_losses.append(reprojection_loss)

                if self.opt.edge_loss:
                    pred_edge = outputs[("edge", frame_id, scale)]
                    dice_score = self.compute_dice_loss(pred_edge, target_edge)
                    dice_loss = 1 - dice_score
                    reprojection_edge = self.compute_reprojection_loss(pred_edge, target_edge)

                    if self.edge_loss_type == "edge_dice":
                        loss += 0.2 * dice_loss.mean()
                        edge_loss_sum_all += dice_loss.mean().detach().item()
                        edge_loss_count_all += 1
                    elif self.edge_loss_type == "edge_ssim":
                        reprojection_losses_edge.append(0.5 * reprojection_edge.float())
                        edge_loss_sum_all += reprojection_edge.mean().detach().item()
                        edge_loss_count_all += 1
                    
            if self.opt.edge_loss:
                reprojection_losses_edge = torch.cat(reprojection_losses_edge, 1)

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                identity_reprojection_losses_edge = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    reprojection_check = self.compute_reprojection_loss(pred, target)
                    identity_reprojection_losses.append(reprojection_check)

                    if self.opt.edge_loss:
                        pred_edge = inputs[("edge", frame_id, source_scale)]
                        dice_check = self.compute_dice_loss(pred_edge, target_edge)
                        dice_loss_check = 1 - dice_check
                        reprojection_edge_check = self.compute_reprojection_loss(pred_edge, target_edge)

                        if self.edge_loss_type == "edge_dice":
                            loss += 0.2 * dice_loss_check.mean()
                            edge_loss_sum_all += dice_loss_check.mean().detach().item()
                            edge_loss_count_all += 1
                        elif self.edge_loss_type == "edge_ssim":
                            identity_reprojection_losses_edge.append(0.5 * reprojection_edge_check.float())
                            edge_loss_sum_all += reprojection_edge_check.mean().detach().item()
                            edge_loss_count_all += 1
                        
                if self.opt.edge_loss:
                    identity_reprojection_losses_edge = torch.cat(identity_reprojection_losses_edge, 1)

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                    if self.opt.edge_loss:
                        # if edge loss is used, we also average the edge reprojection losses
                        identity_reprojection_loss_edge = identity_reprojection_losses_edge.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses
                    if self.opt.edge_loss:
                        identity_reprojection_loss_edge = identity_reprojection_losses_edge

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(mask, [self.opt.height, self.opt.width],
                                        mode="bilinear", align_corners=False)
                reprojection_losses *= mask
                if self.opt.edge_loss:
                    reprojection_losses_edge *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()


            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
                if self.opt.edge_loss:
                    reprojection_losses_edge = reprojection_losses_edge.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses
                if self.opt.edge_loss:
                    reprojection_losses_edge = reprojection_losses_edge

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001
                if self.opt.edge_loss:
                    identity_reprojection_loss_edge += torch.randn(
                        identity_reprojection_loss_edge.shape).cuda() * 0.00001
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
                combined_edge = torch.cat((identity_reprojection_loss_edge, reprojection_losses_edge), dim=1)
            else:
                combined = reprojection_loss
                combined_edge = reprojection_losses_edge

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)
            if self.opt.edge_loss:
                if combined_edge.shape[1] == 1:
                    to_optimise_edge = combined_edge
                else:
                    to_optimise_edge, idxs_edge = torch.min(combined_edge, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()
                if self.opt.edge_loss:
                    outputs["identity_selection_edge/{}".format(scale)] = (
                            idxs_edge > identity_reprojection_loss_edge.shape[1] - 1).float()

            if self.opt.input_mask_path is not None:
                to_optimise = to_optimise[self.input_mask[source_scale]]
                if self.opt.edge_loss:
                    to_optimise_edge = to_optimise_edge[self.input_mask[source_scale]]

            loss += to_optimise.mean()
            if self.opt.edge_loss:
                edge_loss = to_optimise_edge.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color, self.input_mask[scale])

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss
            if self.opt.edge_loss:
                total_loss_edge += edge_loss
                losses["edge_loss/{}".format(scale)] = edge_loss

        total_loss /= self.num_scales
        losses["loss_depth"] = total_loss

        if self.opt.edge_loss:
            total_loss_edge /= self.num_scales
            losses["edge_loss"] = total_loss_edge
            losses["loss_pose"] = total_loss_edge
        else:
            raise ValueError("❌ No edge loss but trying to train pose network. This setup is invalid!")
            # losses["loss_pose"] = torch.tensor(0.0, device=self.device, requires_grad=True)

        # if edge_loss_count_all > 0:
        #     losses["edge_loss"] = edge_loss_sum_all / edge_loss_count_all
        # else:
        #     losses["edge_loss"] = 0.0

        return losses


    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        # torch.save(self.depth_optimizer.state_dict(), save_path.replace(".pth", "_depth.pth"))
        torch.save(self.pose_optimizer.state_dict(), save_path.replace(".pth", "_pose.pth"))


    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            if hasattr(self.opt, "model_load_paths") and n in self.opt.model_load_paths:
                path = self.opt.model_load_paths[n]
                print(f"🔍 Using model_load_paths for {n}: {path}")
            else:
                path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
                print(f"🔍 Using load_weights_folder for {n}: {path}")

            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=self.device)

            adapted_dict = {}
            expected_conv1_keys = [f"{n}.encoder.conv1.weight", f"{n}.conv1.weight"]
            if n == "pose_encoder":
                expected_conv1_keys += ["encoder.conv1.weight"]
            for k, v in pretrained_dict.items():
                if k in expected_conv1_keys:
                    print(f"🔧 Loading {k} weights...")
                    target_in_channels = model_dict[k].shape[1]

                    if n == "pose_encoder":
                        num_input_images = self.num_pose_frames
                        channels_per_image = self.channels_per_image_pose
                    elif n == "encoder":
                        num_input_images = 1
                        channels_per_image = self.channels_per_image_depth
                    else:
                        print(f"⚠️ Unknown model {n} for conv1 reshape")
                        continue

                    print(f"🔁 Adapting {k} from shape {v.shape} to {model_dict[k].shape} "
                        f"(num_input_images={num_input_images}, channels_per_image={channels_per_image})")

                    # pretrained_channels_per_image = v.shape[1] // num_input_images
                    # assert pretrained_channels_per_image == 3, \
                    #     f"Expected 3 channels per image in pretrained weights, got {pretrained_channels_per_image}"

                    pretrained_channels_per_image = v.shape[1] // num_input_images

                    if pretrained_channels_per_image == channels_per_image:
                        print(f"✅ Channels match perfectly: {pretrained_channels_per_image} per image")
                        adapted_dict[k] = v
                        print(f"✅ Loaded {k} without adaptation")
                        continue  

                    elif pretrained_channels_per_image == 3 and channels_per_image > 3:
                        print(f"🔁 Expanding pretrained RGB weights to {channels_per_image} channels per image")

                        rgb_weight = v  # [64, 3*num_input_images, 7, 7]
                        extra_channels = channels_per_image - 3

                        gray = torch.mean(
                            v.reshape(64, num_input_images, 3, 7, 7), dim=2)  # → [64, num_input_images, 7, 7]
                        gray = gray.reshape(64, num_input_images, 1, 7, 7).repeat(1, 1, extra_channels, 1, 1)
                        gray = gray.reshape(64, num_input_images * extra_channels, 7, 7)
                        full_weight = torch.cat([rgb_weight, gray], dim=1)

                    elif pretrained_channels_per_image > 3 and channels_per_image == 3:
                        print(f"🔁 Extracting RGB channels from {pretrained_channels_per_image}-channel pretrained weights")
                        
                        reshaped = v.reshape(64, num_input_images, pretrained_channels_per_image, 7, 7)
                        rgb_only = reshaped[:, :, :3, :, :]  
                        full_weight = rgb_only.reshape(64, num_input_images * 3, 7, 7)

                    else:
                        raise ValueError(
                            f"❌ Cannot adapt pretrained input channels: "
                            f"{pretrained_channels_per_image} → {channels_per_image}"
                        )

                    reordered = []
                    for i in range(num_input_images):
                        for c in range(channels_per_image):
                            reordered.append(full_weight[:, i * channels_per_image + c:i * channels_per_image + c + 1, :, :])
                    new_conv1 = torch.cat(reordered, dim=1)

                    assert new_conv1.shape == model_dict[k].shape, \
                        f"❌ Weight shape mismatch: got {new_conv1.shape}, expected {model_dict[k].shape}"

                    adapted_dict[k] = new_conv1
                    print(f"✅ Adapted {k} to shape {new_conv1.shape}")

                elif k in model_dict and v.shape == model_dict[k].shape:
                    adapted_dict[k] = v
                else:
                    if k in ["height", "width", "use_stereo"]:
                        continue
                    expected = model_dict[k].shape if (
                        k in model_dict and isinstance(model_dict[k], torch.Tensor)
                    ) else type(model_dict.get(k, None))
                    print(f"⚠️ Skipped {k} due to shape mismatch or unsupported type")
            
            print(f"✅ Loaded {len(adapted_dict)} weights into model [{n}]")

            model_dict.update(adapted_dict)
            self.models[n].load_state_dict(model_dict)

        # # loading adam state
        # optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        # if os.path.isfile(optimizer_load_path):
        #     print("Loading Adam weights")
        #     optimizer_dict = torch.load(optimizer_load_path)
        #     self.model_optimizer.load_state_dict(optimizer_dict)
        # else:
        #     print("Cannot find Adam weights so Adam is randomly initialized")



            # if k.endswith("conv1.weight") and v.shape[1] == 3:
            #         print(f"Adapting {k} from shape {v.shape} to {model_dict[k].shape}")
            #         num_input_images = self.num_input_images  # e.g. 1 or 2
            #         rgb_weight = torch.cat([v] * num_input_images, dim=1) / num_input_images
            #         edge_weight = torch.mean(v, dim=1, keepdim=True).repeat(1, num_input_images, 1, 1)

            #         reordered = []
            #         for i in range(num_input_images):
            #             reordered.extend([
            #                 rgb_weight[:, i*3 + 0:i*3 + 1, :, :],  # R
            #                 rgb_weight[:, i*3 + 1:i*3 + 2, :, :],  # G
            #                 rgb_weight[:, i*3 + 2:i*3 + 3, :, :],  # B
            #                 edge_weight[:, i:i+1, :, :]            # E
            #             ])
            #         new_conv1 = torch.cat(reordered, dim=1)
            #         adapted_dict[k] = new_conv1