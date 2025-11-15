# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
import skimage.transform
from evaluate_depth import STEREO_SCALE_FACTOR
import datetime
import csv
import json
import random


def load_monodepth2_model(channels_per_image_depth, channels_per_image_pose):
    import networks
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False, channels_per_image=channels_per_image_depth)
    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))
    return encoder, depth_decoder, None, None

def load_monovit_model_hr(depth_decoder_path, device):
    import networksMonoVIT as networks
    depth_dict = torch.load(depth_decoder_path, map_location=device)
    feed_height = depth_dict['height']
    feed_width = depth_dict['width']
    new_dict = {}
    for k,v in depth_dict.items():
        name = k[7:]
        new_dict[name]=v
    depth_decoder = networks.DeepNet('mpvitnet')
    depth_decoder.load_state_dict({k: v for k, v in new_dict.items() if k in depth_decoder.state_dict()})
    return None, depth_decoder, feed_height, feed_width

def load_monovit_model_lr():
    import networksMonoVIT as networks
    print("   Loading pretrained encoder")
    encoder = networks.mpvit_small()
    encoder.num_ch_enc = [64,128,216,288,288]
    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder()
    return encoder, depth_decoder, None, None

def load_model_safe(method, channels_per_image_depth, channels_per_image_pose, model_path):
    """Safely load model with error handling"""
    try:
        if method == "monodepth":
            return load_monodepth2_model(channels_per_image_depth, channels_per_image_pose)
        elif method == "monovit":
            # Check if this is a high-resolution MonoViT model
            # depth_decoder_path = os.path.join(model_path, "depth.pth")
            # if os.path.exists(depth_decoder_path):
            #     try:
            #         # Try to load as HR model first
            #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #         return load_monovit_model_hr(depth_decoder_path, device)
            #     except:
            #         # Fall back to LR model
            #         print("   HR model loading failed, trying LR model...")
            #         return load_monovit_model_lr()
            # else:
            return load_monovit_model_lr()
        elif method == "IID":
            return load_monodepth2_model(channels_per_image_depth, channels_per_image_pose)
    except Exception as e:
        print(f"   Error loading model: {e}")
        return None, None, None, None

def get_edge(folder_name, frame_index, feed_height, feed_width):
    """Get edge information for a specific frame"""
    filename = f"{folder_name}_{frame_index:03d}.png"
    edge_path = os.path.join('/Datasets/EndoMapper/feast_eval/sample/Edge', filename + '.npy')
    if os.path.exists(edge_path):
        edge = np.load(edge_path)
        edge_resized = skimage.transform.resize(edge, (feed_height, feed_width), order=1, preserve_range=True, mode='constant')
        return torch.tensor(edge_resized, dtype=torch.float32).unsqueeze(0)
    else:
        print(f"Warning: Edge file not found at {edge_path}")
        return torch.zeros((1, feed_height, feed_width), dtype=torch.float32)

def get_lum(folder_name, frame_index, feed_height, feed_width):
    """Get luminance information for a specific frame"""
    filename = f"{folder_name}_{frame_index:03d}.png"
    lum_path = os.path.join(
        '/Datasets/EndoMapper/feast_eval/sample/Shading/IID/finetuned_mono_hkfull_288_pseudo_dsms_automasking_noadjust',
        f"light{filename}")
    if os.path.exists(lum_path):
        lum = pil.open(lum_path)
        lum = np.array(lum, dtype=np.float32)
        lum_resized = skimage.transform.resize(lum, (feed_height, feed_width), order=1, preserve_range=True, mode='constant')
        return torch.tensor(lum_resized, dtype=torch.float32).unsqueeze(0)
    else:
        print(f"Warning: Luminance file not found at {lum_path}")
        return torch.zeros((1, feed_height, feed_width), dtype=torch.float32)

def save_colored_depth(depth_tensor, output_path):
    """Save depth map as colored image"""
    if isinstance(depth_tensor, torch.Tensor):
        depth_np = depth_tensor.squeeze().cpu().numpy()
    else:
        depth_np = depth_tensor
    
    vmax = np.percentile(depth_np, 95)
    vmin = np.percentile(depth_np, 5)
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    mapper = cm.ScalarMappable(norm=normalizer, cmap='jet')
    colormapped_im = (mapper.to_rgba(depth_np)[:, :, :3] * 255).astype(np.uint8)
    
    colored_pil = pil.fromarray(colormapped_im)
    colored_resized = colored_pil.resize((1440, 1080), pil.LANCZOS)
    colored_resized.save(output_path)
    return colored_resized

def process_single_model(model_path, input_dir, output_root, ext='png', target_resolution=(1440, 1080)):
    """Process all images with a single model"""
    
    # Parse model information - use the full model path for better identification
    full_model_name = model_path.replace('/Datasets/', '').replace('/', '_')
    model_name = os.path.basename(model_path)
    
    if 'monodepth' in model_path or 'IID' in model_path:
        method = 'monodepth'
    elif 'monovit' in model_path:
        method = 'monovit'
    else:
        method = 'monodepth'  # default
    
    print(f"\n=== Processing with model: {full_model_name} (method: {method}) ===")
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Create output directory using full model path info
    output_directory = os.path.join(output_root, full_model_name)
    os.makedirs(output_directory, exist_ok=True)
    
    # Create colored depth output directory
    colored_output_directory = os.path.join(output_root, "rb255_"+full_model_name)
    os.makedirs(colored_output_directory, exist_ok=True)
    
    # Determine channels based on full model path (not just basename)
    model_path_lower = model_path.lower()
    if any(k in model_path_lower for k in ["depth_edge", "both_edge", "depth_lum", "both_lum", "dlpe", "depl"]):
        channels_per_image_depth = 4 
    else:
        channels_per_image_depth = 3

    if any(k in model_path_lower for k in ["pose_edge", "both_edge", "pose_lum", "both_lum", "dlpe", "depl"]):
        channels_per_image_pose = 4
    else:
        channels_per_image_pose = 3
    
    print(f"Channels - Depth: {channels_per_image_depth}, Pose: {channels_per_image_pose}")
    
    # Load model
    encoder, depth_decoder, feed_height, feed_width = load_model_safe(
        method, channels_per_image_depth, channels_per_image_pose, model_path)
    
    if encoder is None and depth_decoder is None:
        print(f"Failed to load model from {model_path}")
        return
    
    depth_decoder_path = os.path.join(model_path, "depth.pth")
    
    if encoder is not None:
        encoder_path = os.path.join(model_path, "encoder.pth")
        if not os.path.exists(encoder_path):
            print(f"Encoder file not found: {encoder_path}")
            return
            
        try:
            loaded_dict_enc = torch.load(encoder_path, map_location=device)
            
            feed_height = loaded_dict_enc['height']
            feed_width = loaded_dict_enc['width']
            
            # Filter the state dict to match the model architecture
            filtered_dict_enc = {}
            for k, v in loaded_dict_enc.items():
                if k in encoder.state_dict():
                    if encoder.state_dict()[k].shape == v.shape:
                        filtered_dict_enc[k] = v
                    else:
                        print(f"Shape mismatch for {k}: model={encoder.state_dict()[k].shape}, checkpoint={v.shape}")
            
            encoder.load_state_dict(filtered_dict_enc, strict=False)
            encoder.to(device)
            encoder.eval()
            
            if os.path.exists(depth_decoder_path):
                loaded_dict = torch.load(depth_decoder_path, map_location=device)
                depth_decoder.load_state_dict(loaded_dict, strict=False)
        except Exception as e:
            print(f"Error loading encoder/decoder: {e}")
            return
    else:
        # For MonoViT HR models, model is already loaded
        if feed_height is None or feed_width is None:
            feed_height, feed_width = 288, 288  # default values
    
    depth_decoder.to(device)
    depth_decoder.eval()
    
    # Get all image files from the input directory
    image_paths = glob.glob(os.path.join(input_dir, f'*.{ext}'))
    
    print(f"-> Found {len(image_paths)} images to process")
    print(f"-> Target output resolution: {target_resolution[0]}x{target_resolution[1]}")
    
    with torch.no_grad():
        for idx, image_path in enumerate(image_paths):
            if image_path.endswith(f"_disp.{ext}"):
                continue
            
            print(f"Processing {idx+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            # Load and preprocess image
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            rgb_tensor = transforms.ToTensor()(input_image_resized)
            
            # Parse filename to extract folder_name and frame_index
            filename = os.path.basename(image_path)
            # Assuming format: HCULB_00033_procedure_lossy_h264_003.png
            parts = filename.split('_')
            if len(parts) >= 5:
                folder_name = '_'.join(parts[:-1])  # Everything except the last part
                frame_index = int(parts[-1].split('.')[0])  # Last number before extension
            else:
                # Fallback parsing
                base_name = os.path.splitext(filename)[0]
                parts = base_name.split('_')
                frame_index = int(parts[-1])
                folder_name = '_'.join(parts[:-1])
            
            # Check if we need edge or luminance inputs
            use_edge = any(k in model_path_lower for k in ["depth_edge", "both_edge", "depl"])
            use_lum = any(k in model_path_lower for k in ["depth_lum", "both_lum", "dlpe"])
            
            # Prepare input tensor
            extra_inputs = [rgb_tensor]
            if use_edge:
                edge_tensor = get_edge(folder_name, frame_index, feed_height, feed_width)
                extra_inputs.append(edge_tensor)
            if use_lum:
                lum_tensor = get_lum(folder_name, frame_index, feed_height, feed_width)
                extra_inputs.append(lum_tensor)
            
            input_tensor = torch.cat(extra_inputs, dim=0).unsqueeze(0).to(device).to(torch.float32)
            
            print(f"Input tensor shape: {input_tensor.shape}, min: {input_tensor.min().item():.4f}, max: {input_tensor.max().item():.4f}")
            
            # Forward pass
            try:
                if encoder is not None:
                    features = encoder(input_tensor)
                    outputs = depth_decoder(features)
                else:
                    outputs = depth_decoder(input_tensor)
            except Exception as e:
                print(f"Error during forward pass: {e}")
                continue
            
            # Convert disparity to depth
            pred_disp, _ = disp_to_depth(outputs[("disp", 0)], 0.1, 100.0)  # min_depth=0.1, max_depth=100.0
            
            print(f"pred_disp shape: {pred_disp.shape}, min: {torch.min(pred_disp):.4f}, max: {torch.max(pred_disp):.4f}")
            
            # Resize to target resolution instead of original image size
            disp_resized = torch.nn.functional.interpolate(
                pred_disp, target_resolution[::-1], mode="bilinear", align_corners=False)  # Note: target_resolution is (width, height), but interpolate expects (height, width)
            
            pred_depth = (1/disp_resized).squeeze().cpu().numpy()
            print(f"Pred Depth - min: {np.min(pred_depth):.4f}, max: {np.max(pred_depth):.4f}, mean: {np.mean(pred_depth):.4f}")
            
            # Save grayscale depth map
            max_value = np.max(pred_depth)
            trip_im = pil.fromarray(np.stack((pred_depth*255/max_value,)*3, axis=-1).astype(np.uint8))
            
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_directory, f"{output_name}.{ext}")
            trip_im.save(output_path)
            print(f"Saved grayscale: {output_path}")
            
            # Save colored depth map
            colored_output_path = os.path.join(colored_output_directory, f"{output_name}_colored.{ext}")
            save_colored_depth(pred_depth, colored_output_path)
            print(f"Saved colored: {colored_output_path}")

def main():
    # Define paths
    input_dir = "/Datasets/EndoMapper/feast_eval/sample/Resized/to_eval"
    output_root = "/Datasets/EndoMapper/feast_eval/sample/Depth_Output"
    
    # Define model list
    model_list = [
        # "/Datasets/Weights_Baselines_finetune/monodepth/hkfull_mono_finetuned/models/weights_19",
        # "/Datasets/Weights_Baselines_finetune/monodepth/c3vd_mono_finetuned/models/weights_19",
        # "/Datasets/Weights_Baselines_finetune/monovit/hkfull_mono_finetuned/models/weights_19",
        # "/Datasets/Weights_Baselines_finetune/monovit/c3vd_mono_finetuned/models/weights_19",
        # "/Datasets/Weights_Baselines_finetune/monovit/c3vd_mysplit_interval30/models/weights_19",
        # "/Datasets/weights/Baselines/AF-sfm/c3vd_w_pretrained/models/weights_19",
        # "/Datasets/weights/Baselines/AF-sfm/c3vd30_w_pretrained/models/weights_19",
        # '/Datasets/weights/Baselines/AF-sfm/hk_w_pretrained/models/weights_19',
        # "/Datasets/weights/Baselines/iid-sfm/c3vd_w_pretrained/models/weights_19",
        # "/Datasets/weights/Baselines/iid-sfm/c3vd30_w_pretrained/models/weights_19",
        # '/Datasets/weights/Baselines/iid-sfm/hk_w_pretrained/models/weights_19',
        # '/Datasets/weights/xinwei/IID_finetuned_mono_hkfull_288_pseudo_dsms_automasking_noadjust/weights_19',
        "/Datasets/Weights_Feast_new/monodepth/hk_mono_finetuned_dlpe_edge_ssim_depth_fix/models/weights_29",
        # "/Datasets/Weights_Feast_new/monodepth/hk_mono_finetuned_both_lum_edge_ssim_depth_fix/models/weights_19",
        # "/Datasets/Weights_Feast_all/monodepth/hk_mono_finetuned_both_lum/models/weights_19"
    ]
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_root}")
    print(f"Processing {len(model_list)} models")
    print(f"Target resolution: 1440x1080")
    
    # Process each model
    for i, model_path in enumerate(model_list):
        print(f"\n{'='*60}")
        print(f"Processing model {i+1}/{len(model_list)}: {model_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(model_path):
            print(f"WARNING: Model path does not exist: {model_path}")
            continue
            
        try:
            process_single_model(model_path, input_dir, output_root, target_resolution=(288, 288))
            print(f"Successfully completed model {i+1}/{len(model_list)}")
        except Exception as e:
            print(f"ERROR processing model {model_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("All models processed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()