# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse
from pyexpat import model
import numpy as np
import torch
from torch.utils.data import DataLoader
from layers import transformation_from_parameters
from utils import readlines
import networks
from scipy.spatial.transform import Rotation as R, Slerp
from depth_evaluate_max_norm import load_model  



# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def to_cuda(obj):
    if torch.is_tensor(obj):
        return obj.cuda()
    elif isinstance(obj, list):
        return [to_cuda(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: to_cuda(v) for k, v in obj.items()}
    else:
        return obj


def interpolate_frame_to_frame_corrected(T_rel, num_interpolations):
    R_rel = T_rel[:3, :3]
    t_rel = T_rel[:3, 3]
    q_rel = R.from_matrix(R_rel).as_quat()
    slerp = Slerp([0, 1], R.from_quat([[1, 0, 0, 0], q_rel]))
    timestamps = np.linspace(0, 1, num_interpolations + 1)
    global_rotations = slerp(timestamps).as_matrix()
    global_translations = [t_rel * alpha for alpha in timestamps]
    transforms = []
    for i in range(num_interpolations):
        R_step = global_rotations[i].T @ global_rotations[i + 1]
        t_step = global_translations[i + 1] - global_translations[i]
        T = np.eye(4)
        T[:3, :3] = R_step
        T[:3, 3] = t_step
        transforms.append(T)
    return transforms

def load_monodepth2_model(channels_per_image_depth, channels_per_image_pose, method = "monodepth"):
    if method == "monodepth":
        import networks
        print("   Loading pretrained encoder")
        encoder = networks.ResnetEncoder(18, True, num_input_images=2, channels_per_image=channels_per_image_pose)
        print("   Loading pretrained decoder")
        pose_decoder = networks.PoseDecoder(
            num_ch_enc=encoder.num_ch_enc,
                        num_input_features=1,
                        num_frames_to_predict_for=2)
    elif method == "IID":
        import networksIID
        print("   Loading IID pretrained encoder")
        encoder = networksIID.ResnetEncoder(18, True, num_input_images=2)
        print("   Loading IID pretrained decoder")
        pose_decoder = networksIID.PoseDecoder(
            num_ch_enc=encoder.num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)
    return encoder, pose_decoder

def load_model(channels_per_image_depth, channels_per_image_pose, method="monodepth"):
    return load_monodepth2_model(channels_per_image_depth, channels_per_image_pose, method=method)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model folder under /Datasets/Weights_Feast_compare/")
    parser.add_argument("--weights_base", type=str, default="/Datasets/Weights_Feast_compare",
                        help="Base path to model weights")
    parser.add_argument("--output_base", type=str, default="/Datasets/Weights_Feast_compare/Pose_Output",
                        help="Base path for saving pose predictions")
    parser.add_argument("--data_path", type=str, default="/raid/xinwei/dataset/c3vd/registered_sequences",
                        help="Path to the C3VD dataset")
    parser.add_argument("--split_file_root", type=str, default="splits/c3vd/test_files_interval{}.txt",
                        help="Path format for test file list")
    parser.add_argument("--num_interpolations", type=int, default=10,
                        help="Number of intermediate frames to interpolate")
    parser.add_argument("--method", type=str, default="monodepth")
    return parser.parse_args()


def run_pose_prediction_on_video(args, video_name):
    import PIL.Image as Image
    from torchvision import transforms

    print(f"Method: {args.method}")

    if any(k in args.model_name for k in ["pose_edge", "both_edge", "pose_lum", "both_lum", "dlpe", "depl"]):
        channels_per_image_pose = 4
    else:
        channels_per_image_pose = 3

    pose_encoder, pose_decoder = load_model(
        channels_per_image_depth=3,
        channels_per_image_pose=channels_per_image_pose,
        method=args.method
    )

    load_weights_folder = os.path.join(args.weights_base, args.model_name) #
    pose_encoder.load_state_dict(torch.load(os.path.join(load_weights_folder, "models/weights_19", "pose_encoder.pth")))
    pose_decoder.load_state_dict(torch.load(os.path.join(load_weights_folder, "models/weights_19", "pose.pth")))
    pose_encoder.cuda().eval()
    pose_decoder.cuda().eval()

    interval_str = [s for s in args.model_name.split("_") if "interval" in s]
    if len(interval_str) > 0:
        interval = int(interval_str[0].replace("interval", ""))
    else:
        interval = 1  # default
    print(f"[{args.model_name}] Using interval = {interval}")


    def load_image(folder, idx):
        image_path = os.path.join(args.data_path, folder, f"{idx:04d}_color.png")
        image = Image.open(image_path).convert('RGB').resize((288, 288))
        return transforms.ToTensor()(image)

    def get_edge(folder, idx):
        path = os.path.join('/Datasets/C3VD_Undistorted/Edge', folder, 'avg', f"{idx:04d}_color.png.npy")
        edge = np.load(path)
        return torch.tensor(edge, dtype=torch.float32).unsqueeze(0)

    def get_lum(folder, idx):
        path = os.path.join('/Datasets/C3VD_Undistorted/Shading/models/weights_19', folder, 'decomposed', f"light{idx:04d}_color.png")
        lum = np.array(Image.open(path)).astype(np.float32)
        return torch.tensor(lum, dtype=torch.float32).unsqueeze(0)

    import glob

    # 获取该视频中所有 frame 文件数量
    image_dir = os.path.join(args.data_path, video_name)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*_color.png")))
    max_index = len(image_paths)
    print(f"Video {video_name} has {max_index} frames")

    for i in range(max_index - interval):
        frames = []
        for offset in [0, interval]:
            idx = i + offset
            img = load_image(video_name, idx)
            extras = []
            if "pose_edge" in args.model_name or "both_edge" in args.model_name or "dlpe" in args.model_name:
                extras.append(get_edge(video_name, idx))
            if "pose_lum" in args.model_name or "both_lum" in args.model_name or "depl" in args.model_name:
                extras.append(get_lum(video_name, idx))
            if extras:
                img = torch.cat([img] + extras, dim=0)
            frames.append(img.unsqueeze(0))  # (1, C, H, W)

        input_pair = torch.cat(frames, dim=1).cuda().float()

        with torch.no_grad():
            features = [pose_encoder(input_pair)]
            axisangle, translation = pose_decoder(features)
            predicted_pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0])
            predicted_pose_np = predicted_pose.cpu().numpy().squeeze(0)

        out_file = f"{i:04d}_color_to_{i + interval:04d}_color.txt"
        output_dir = os.path.join(args.output_base, args.model_name, video_name)
        os.makedirs(output_dir, exist_ok=True)
        path_to_save = os.path.join(output_dir, out_file)
        with open(path_to_save, "w") as f:
            f.write(" ".join(map(str, predicted_pose_np.flatten())))
        print("Saved:", path_to_save)



if __name__ == "__main__":
    base_model_path = "/Datasets"

    # model_list = sorted([
    #     m for m in os.listdir(base_model_path)
    #     if os.path.isdir(os.path.join(base_model_path, m))
    #     and m not in ["Depth_Output", "Pose_Output"]
    # ])
    # model_list = ['FEAST_Weights/Weights_Feast_new/monodepth_depth_fix_thorough_pose_from_scratch/hk_mono_finetuned_dlpe_edge_ssim/models/weights_19',
                #   'FEAST_Weights/Weights_Feast_Dataset_Finetune_all/AF-sfm/hk_w_pretrained/models/weights_19',
                #   'FEAST_Weights/Weights_Feast_Dataset_Finetune_all/iid-sfm/hk_w_pretrained/models/weights_19', 
                #   '/Datasets/Checkpoints/af-sfm',
                #   '/Datasets/Checkpoints/iid-sfm'
                #   ]

    model_list = ["/Datasets/Weights_FEAST/Weights_Feast_Dataset_Finetune_all/monodepth/c3vd_mono_finetuned",
                  "/Datasets/Weights_FEAST/Weights_Feast_Dataset_Finetune_all/monodepth/c3vd_mysplit_interval10"]

    # model_list = ["Weights_FEAST/Weights_Feast_Dataset_Finetune_all/AF-sfm/c3vd_w_pretrained",
    #               "/Datasets/Weights_FEAST/Weights_Feast_Dataset_Finetune_all/iid-sfm/c3vd_w_pretrained",
    #               "/Datasets/Weights_FEAST/Weights_Feast_Dataset_Finetune_all/monodepth/c3vd_mysplit_finetuned",
    #               "/Datasets/Weights_FEAST/Weights_Feast_Dataset_Finetune_all/monovit/c3vd_mysplit_finetuned"]
    
    print(f"Model list: {model_list}")

    video_list = ["trans_t4_b", "sigmoid_t3_b", "desc_t4_a", "cecum_t4_b", "cecum_t1_a"]

    for model_name in model_list:
        print(f"\n===> Running: {model_name}")
        import argparse

        args = argparse.Namespace(
            model_name=model_name,
            weights_base=base_model_path,
            output_base='/Workspace/KeyFrameGraph/Outputs',
            data_path="/Datasets/C3VD_Undistorted/Dataset",
            # split_file_root="splits/c3vd/test_files_interval{}.txt",
            method = "monodepth",
        )

        for video in video_list:
            run_pose_prediction_on_video(args, video)
