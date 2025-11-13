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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 确保确定性
    torch.backends.cudnn.benchmark = False  # 关闭 CuDNN 优化，保证一致性

set_seed(42)

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    ## change all value from 16 bit to 0-100 mm
    gt = gt * 100 / (2**16-1)
    pred = pred * 100 / (2**16-1)
    # print("gt", gt.shape, gt[0,0], np.min(gt), np.max(gt))
    # Root Mean Squared Error
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    # Root Mean Squared Logarithmic Error
    rmse_log = (np.log(gt + 1e-6) - np.log(pred + 1e-6)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    # Mean Absolute Error
    mae = np.mean(np.abs(gt - pred))
    # Median Absolute Error
    medae = np.median(np.abs(gt - pred))
    # Absolute Relative Error
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    # Squared Relative Error
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    
    # Sigma accuracy metrics
    thresh = np.maximum((gt / pred), (pred / gt))
    sigma1 = np.mean(thresh < 1.25)
    sigma2 = np.mean(thresh < 1.25 ** 2)
    sigma3 = np.mean(thresh < 1.25 ** 3)
    
    return rmse, rmse_log, mae, medae, abs_rel, sq_rel, sigma1, sigma2, sigma3

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
    #new_dict = depth_dict
    new_dict = {}
    for k,v in depth_dict.items():
        name = k[7:]
        new_dict[name]=v
    depth_decoder = networks.DeepNet('mpvitnet')
    depth_decoder.load_state_dict({k: v for k, v in new_dict.items() if k in depth_decoder.state_dict()}, strict=False)
    return None, depth_decoder, feed_height, feed_width

def load_monovit_model_lr():
    import networksMonoVIT as networks
    print("   Loading pretrained encoder")
    encoder = networks.mpvit_small() #networks.ResnetEncoder(opt.num_layers, False)
    encoder.num_ch_enc = [64,128,216,288,288]  # = networks.ResnetEncoder(opt.num_layers, False)
    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder()
    return encoder, depth_decoder, None, None

def load_model(method, channels_per_image_depth, channels_per_image_pose):
    if method == "monodepth":
        return load_monodepth2_model(channels_per_image_depth, channels_per_image_pose)
    elif method == "monovit":
        # if "640" not in model_name and "288" not in model_name:
        #     return load_monovit_model_hr(depth_decoder_path, device)
        # else:
        return load_monovit_model_lr()
            

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for MonoVIT/Monodepth2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use')
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')
    parser.add_argument("--method", type=str,
                        help='method to use for depth prediction',
                        choices=["monodepth", "monovit"],
                        default="monovit")
    parser.add_argument("--output_path", type=str,
                        help='output path for saving predictions in a folder',
                        default="output")
    parser.add_argument("--model_basepath", type=str,
                        help='base path for model files',
                        default="models")
    parser.add_argument("--config", type=str,
                        help='config file with training parameters',
                        default=None)
    parser.add_argument("--save_depth", action='store_true',
                        help='save predicted depth maps')
    parser.add_argument("--eval", action='store_true',
                        help='evaluate predicted depth maps')
    parser.add_argument("--min_depth", type=float, default=0.1,
                        help='minimum depth for evaluation')
    parser.add_argument("--max_depth", type=float, default=100, 
                        help='maximum depth for evaluation')
    parser.add_argument("--seq", type=str, nargs='+', default=[""],
                        help='sequence(s) to evaluate')
    parser.add_argument("--save_triplet", action='store_true',
                        help='save image depth and ground truth')
    parser.add_argument("--disable_median_scaling", action='store_true',
                        help='disable median scaling')
    parser.add_argument("--pred_depth_scale_factor", type=float, default=1.0,
                        help='depth prediction scaling factor')
    parser.add_argument("--median_scaling_specular", action='store_true',
                        help='use median scaling only on specular pixels')
    parser.add_argument("--notclipped", action='store_true',
                        help='skip clipping depth values')
    parser.add_argument("--input_mask", type=str, default=None,)

    return parser.parse_args()


def test_simple(args, seq):
    # the c3vd dataset follows the format discribed bellow
    # depthmaps are stored as uint_16 .tiff
    # values between 0-(2**16-1) are map to 0-100 mm
    MIN_DEPTH = 0
    MAX_DEPTH = (2**16-1)
    
    """Function to predict for a single image or folder of images
    """
    errors = []
    errors_masked = []
    ratios = []
    
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.pred_metric_depth and "stereo" not in args.model_name:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")
    
    model_path = os.path.join(args.model_basepath, args.model_name)
    print("evaluating model: ", model_path)

    # download_model_if_doesnt_exist(args.model_basepath, args.method, args.model_name)


    print("-> Loading model from ", model_path)

    depth_decoder_path = os.path.join(model_path, "depth.pth")
    
    # LOADING PRETRAINED MODEL
    print("model_name", args.model_name)

    if "depth_edge" in args.model_name or "both_edge" in args.model_name or "depth_lum" in args.model_name or "both_lum" in args.model_name or "dlpe" in args.model_name or "depl" in args.model_name:
        channels_per_image_depth = 4 
    else:
        channels_per_image_depth = 3

    if "pose_edge" in args.model_name or "both_edge" in args.model_name or "pose_lum" in args.model_name or "both_lum" in args.model_name or "dlpe" in args.model_name or "depl" in args.model_name:
        channels_per_image_pose = 4
    else:
        channels_per_image_pose = 3

    encoder, depth_decoder, feed_height, feed_width = load_model(
        args.method, channels_per_image_depth, channels_per_image_pose)

    if encoder is not None:
        encoder_path = os.path.join(model_path, "encoder.pth")
        loaded_dict_enc = torch.load(encoder_path, map_location=device)

        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)
        encoder.to(device)
        encoder.eval()

        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        depth_decoder.load_state_dict(loaded_dict, strict=False)

    depth_decoder.to(device)
    depth_decoder.eval()

    if os.path.isfile(args.image_path):
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        paths = glob.glob(os.path.join(args.image_path, seq, '*.{}'.format(args.ext)))
        output_directory = os.path.join(args.output_path, args.model_name, args.type_data, seq)
        os.makedirs(output_directory, exist_ok=True)
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    def get_edge(folder, frame_index):
        edge_path = os.path.join(
            '/Datasets/C3VD_Undistorted/Edge', folder, 'avg', f"{frame_index:04d}_color.png.npy")
        edge = np.load(edge_path)
        edge_resized = skimage.transform.resize(edge, (feed_height, feed_width), order=1, preserve_range=True, mode='constant')
        return torch.tensor(edge_resized, dtype=torch.float32).unsqueeze(0)

    def get_lum(folder, frame_index):
        lum_path = os.path.join(
            '/Datasets/C3VD_Undistorted/Shading/models/weights_19',
            folder, 'decomposed', f"light{frame_index:04d}_color.png")
        lum = pil.open(lum_path)
        lum = np.array(lum, dtype=np.float32)
        return torch.tensor(lum, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            if image_path.endswith("_disp.{}".format(args.ext)):
                continue

            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            rgb_tensor = transforms.ToTensor()(input_image_resized)

            folder_path = os.path.dirname(image_path)
            folder_name = os.path.basename(folder_path)
            filename = os.path.basename(image_path)
            frame_index = int(filename.split("_")[0])

            use_edge = any(k in args.model_name for k in ["depth_edge", "both_edge", "depl"])
            use_lum = any(k in args.model_name for k in ["depth_lum", "both_lum", "dlpe"])

            extra_inputs = [rgb_tensor]
            if use_edge:
                edge_tensor = get_edge(folder_name, frame_index)
                extra_inputs.append(edge_tensor)
            if use_lum:
                lum_tensor = get_lum(folder_name, frame_index)
                extra_inputs.append(lum_tensor)

            input_tensor = torch.cat(extra_inputs, dim=0).unsqueeze(0).to(device).to(torch.float32)

            print("Input tensor shape:", input_tensor.shape,
                "min:", input_tensor.min().item(),
                "max:", input_tensor.max().item())

            features = encoder(input_tensor)
            outputs = depth_decoder(features)


            
            pred_disp, _ = disp_to_depth(outputs[("disp", 0)], args.min_depth, args.max_depth)

            print("pred_disp", pred_disp.shape,
                torch.min(pred_disp), torch.max(pred_disp))

            disp_resized = torch.nn.functional.interpolate(
                pred_disp, (original_height, original_width), mode="bilinear", align_corners=False)
            
            # print("disp_resized", disp_resized.shape, disp_resized[0,0,0,0], 
            #     torch.min(disp_resized), torch.max(disp_resized))
            pred_depth = (1/disp_resized).squeeze().cpu().numpy()
            print("Pred Depth - min:", np.min(pred_depth), "max:", np.max(pred_depth), "mean:", np.mean(pred_depth))


            
            if args.save_depth:
                # mask = np.array(pil.open("/media/rema/data/DataHKGab/mask_hk_288.png").convert('L')) > 0 
                max_value = np.max(pred_depth)
                trip_im = pil.fromarray(np.stack((pred_depth*255/max_value,)*3, axis=-1).astype(np.uint8))
                # trip_im.save("outputimage.png")
                # if args.input_mask is not None:
                #     print("input_mask", type(trip_im), type(input_mask_pil))
                #     print("input_mask", np.array(trip_im).shape, np.array(input_mask_pil).shape, (np.array(input_mask_pil)/255).shape, np.array(input_mask_pil).max())
                #     mask_1 = (np.array(input_mask_pil)/255).astype(np.uint8)
                #     pil.fromarray((np.array(trip_im)*mask_1)).save("outputimage_masked.png")
                #     trip_im.save("outputimage_from_masked.png")
                # exit(0)
            
                output_name_trip = os.path.splitext(os.path.basename(image_path))[0]
                name_dest_im_trip = os.path.join(output_directory, "{}.{}".format(output_name_trip, args.ext))
                trip_im.save(name_dest_im_trip)
                print(name_dest_im_trip)            
            
                        
            if args.eval:
                # load gt 
                tiff_gt_depth = pil.open(image_path.replace("color", "depth").replace("png", "tiff"))
                gt_depth = np.array(tiff_gt_depth, dtype=np.float32)
                
                
                # # spec mask
                # spec_mask = pil.open(image_path.rsplit("Dataset", 1)[0] + "Annotations_Dilated" + image_path.rsplit("Dataset", 1)[1])
                # spec_mask = spec_mask.convert('L')
                # spec_mask = spec_mask.point( lambda p: 255 if p > 200 else 0 )
                # spec_mask = np.array(spec_mask.convert('1'))


                # c3vd mask
                
                mask = gt_depth > 0
                
                
                if not args.disable_median_scaling:
                    ## uncomment below if evaluate spec mask model
                    # if args.median_scaling_specular and spec_mask.sum() > 0:
                    #     ratio = np.median(gt_depth[spec_mask*mask]) / np.median(pred_depth[spec_mask*mask])
                    # else:
                    ratio = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
                    ratios.append(ratio)
                    pred_depth *= ratio
                else:
                    # to get this value monodepth runs this once and gets the median of the ratios as the single scale factor
                    pred_depth *= args.pred_depth_scale_factor

                # save triplet                
                if args.save_triplet:
                    max_value = np.max([np.max(gt_depth), np.max(pred_depth)])
                    trip_im = pil.fromarray(np.hstack([np.array(pil.open(image_path).convert('RGB')),
                                                       np.stack((pred_depth*mask*255/max_value,)*3, axis=-1).astype(np.uint8), 
                                                       np.stack((gt_depth*mask*255/max_value,)*3, axis=-1).astype(np.uint8)
                                                       ]))
                    output_name_trip = os.path.splitext(os.path.basename(image_path))[0]
                    name_dest_im_trip = os.path.join(output_directory, "{}_triplet.{}".format(output_name_trip, args.ext))
                    trip_im.save(name_dest_im_trip)
                    # print(name_dest_im_trip)
                    
                    
                pred_depth_masked = pred_depth[mask]
                gt_depth_masked = gt_depth[mask]
                
                if not args.notclipped:
                    pred_depth_masked[pred_depth_masked < MIN_DEPTH] = MIN_DEPTH
                    pred_depth_masked[pred_depth_masked > MAX_DEPTH] = MAX_DEPTH
                
                print("GT Depth - min:", np.min(gt_depth), "max:", np.max(gt_depth), "mean:", np.mean(gt_depth))
                print("Pred Depth - min:", np.min(pred_depth), "max:", np.max(pred_depth), "mean:", np.mean(pred_depth))
                print("gt_depth_masked", gt_depth_masked.shape, np.min(gt_depth_masked), np.max(gt_depth_masked))
                print("pred_depth_masked", pred_depth_masked.shape, np.min(pred_depth_masked), np.max(pred_depth_masked))
                
                errors.append(compute_errors(gt_depth_masked, pred_depth_masked))
                
                ## uncomment below if evaluate spec mask model
                # if np.unique(spec_mask*mask).shape[0] == 2:
                #     pred_depth_spec_masked = pred_depth[spec_mask*mask]
                #     gt_depth_spec_masked = gt_depth[spec_mask*mask]
                    
                #     if not args.notclipped:
                #         pred_depth_spec_masked[pred_depth_spec_masked < MIN_DEPTH] = MIN_DEPTH
                #         pred_depth_spec_masked[pred_depth_spec_masked > MAX_DEPTH] = MAX_DEPTH
                    
                    
                #     errors_masked.append(compute_errors(gt_depth_spec_masked, pred_depth_spec_masked))
                # else:
                #     print("No valid pixels in spec_mask")
    
    if args.eval:
        if not args.disable_median_scaling:
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
    
        rmse, rmse_log, mae, medae, abs_rel, sq_rel, sigma1, sigma2, sigma3 = np.array(errors).mean(0) 
        # mean_errors = np.array(errors).mean(0)
        # mean_errors_masked = np.array(errors_masked).mean(0)
        
        return rmse, rmse_log, mae, medae, abs_rel, sq_rel, sigma1, sigma2, sigma3  #mean_errors#, mean_errors_masked
    else:
        return None, None
        


if __name__ == '__main__':
    args = parse_args()

    if args.config is not None:
        # Load args from the configuration file
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Update default args with args from the configuration file
        for key, value in config.items():
            setattr(args, key, value)

    if args.eval:
        date = datetime.datetime.now().strftime("_%Y-%m-%d")
        out = os.path.join(args.output_path, args.model_name)
        os.makedirs(out, exist_ok=True)
        if args.notclipped:
            notclipped = "_notclipped"
        else: 
            notclipped = ""
        file =  open(f"{out}results{notclipped}{date}.csv", mode='w')
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['video', 'rmse', 'rmse_log', 'mae', 'medae', 'abs_rel', 'sq_rel', 'sigma1', 'sigma2', 'sigma3'])
    
    args.type_data = ""
    # Check if args.seq is set to 'all'
    if os.path.isdir(args.image_path) and args.seq == ['all']:
        # List all folders in args.image_path
        sequences = sorted([folder for folder in os.listdir(args.image_path) if os.path.isdir(os.path.join(args.image_path, folder))])
    elif os.path.isdir(args.image_path):
        sequences = args.seq
    ## single image path changed !!
    # elif os.path.isfile(args.image_path) and args.image_path.endswith('.txt'):
    #     if args.image_path.endswith('inpainted.txt'):
    #         args.type_data = "hkinpainted"
    #     else:
    #         args.type_data = "hk"
    #     # Read list of image directories:
    #     with open(args.image_path) as f:
    #         images = f.read().splitlines()
    #     # Extract unique last folder names directly
    #     sequences = list({os.path.basename(os.path.normpath(os.path.split(path)[0])) for path in images})
    #     args.image_path = os.path.dirname(os.path.dirname(images[0]))

    elif os.path.isfile(args.image_path):
        sequences = ["single"]  # ✅ 修复：为单图像路径创建一个 dummy sequence
    else:
        raise Exception(f"Invalid image_path: {args.image_path}")


    for seq in sequences:
        rmse, rmse_log, mae, medae, abs_rel, sq_rel, sigma1, sigma2, sigma3 = test_simple(args, seq)
        # print(f"Sequence {seq}:")
        ## uncomment below if evaluate spec mask model
        # mean_errors, mean_errors_masked = test_simple(args, seq)
        # save results to csv using unique_dirs
        if args.eval:
            writer.writerow([seq, rmse, rmse_log, mae, medae, abs_rel, sq_rel, sigma1, sigma2, sigma3])

    if args.eval:
        file.close()