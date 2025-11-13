from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
from .mono_dataset import MonoDataset
import skimage.transform
import torch

class HKInitDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, distorted=False, **kwargs):
        super(HKInitDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        # HK from MeanIntrinsics.ipynb for C3VD data: 
        if distorted:
            intrinsics = [0.7428879629629629, 0.7424861111111111, 0.4937833333333333, 0.5071601851851851]
        else:
            intrinsics = [0.6423119966210151, 0.6401273085242651, 0.4824200466376491, 0.5298353978680292]
        
        # should i assume 0.5 0.5 center????? no i will remove flip
        # also add c3vd option
        self.K = np.array([[intrinsics[0], 0, intrinsics[2], 0],
                           [0, intrinsics[1], intrinsics[3], 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (288, 288)

    # def check_depth(self):
    #     velo_filename = r"velodyne_points/data/"
    #     return os.path.isfile(velo_filename)
    
    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        # if do_flip:
        #     color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    

class HKDataset(HKInitDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(HKDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        if "C3VD" in folder:
            f_str = "{:04d}_color{}".format(frame_index, self.img_ext)
        elif "BBPS-2-3Frames" in folder:
            f_str = "{:05d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(folder, f_str)
        return image_path
    
    def get_depth(self, folder, frame_index, side, do_flip):
        if "C3VD" in folder:
            f_str = "{:04d}_depth{}".format(frame_index, '.tiff')

        depth_16bit = pil.open(os.path.join(folder, f_str))
        depth_gt = np.array(depth_16bit, dtype=np.float32) # this is 16bit depth
        depth_gt = depth_gt / (2**16-1) 
        return depth_gt
    
    def get_edge(self, folder, frame_index):
        # print("folder", folder)
        edge_path = os.path.join(
            '/Datasets/C3VD_Undistorted/Edge',
            # '/Datasets/Hyper-Kvasir/BBPS-2-3Undistorted_Edge',
            folder,
            'avg',
            f"{frame_index:04d}_color.png.npy")
        # print("edge_path", edge_path)
        edge = np.load(edge_path)  

        edge_resized = skimage.transform.resize(
            edge, (self.height, self.width), order=1, preserve_range=True, mode='constant')
        edge_tensor = torch.tensor(edge_resized, dtype=torch.float32).unsqueeze(0)
        return edge_tensor
    
    def get_edge_hk(self, folder, frame_index):
        # print("folder", folder)
        edge_path = os.path.join(
            '/Datasets/Hyper-Kvasir/BBPS-2-3Undistorted_Edge',
            folder,
            'avg',
            f"{frame_index:05d}.png.npy")
        # print("edge_path", edge_path)
        edge = np.load(edge_path)  

        edge_resized = skimage.transform.resize(
            edge, (self.height, self.width), order=1, preserve_range=True, mode='constant')
        edge_tensor = torch.tensor(edge_resized, dtype=torch.float32).unsqueeze(0)
        return edge_tensor
    
    def get_lum(self, folder, frame_index):
        # print("folder", folder)
        lum_path = os.path.join(
            '/Datasets/C3VD_Undistorted/Shading/models/weights_19',
            folder,
            'decomposed',
            f"light{frame_index:04d}_color.png")
        ## load png
        # print("lum_path", lum_path)
        lum = pil.open(lum_path)
        lum = np.array(lum, dtype=np.float32) # this is 16bit depth

        lum_tensor = torch.tensor(lum, dtype=torch.float32).unsqueeze(0)
        return lum_tensor
    
    def get_lum_hk(self, folder, frame_index):
        # print("folder", folder)
        lum_path = os.path.join(
            '/Datasets/Hyper-Kvasir/BBPS-2-3Undistorted_Shading/models/weights_19',  ## IID/weights_19 should be used for lum_reflect training
            folder,
            'decomposed',
            f"light{frame_index:05d}.png")
        ## load png
        # print("lum_path", lum_path)
        lum = pil.open(lum_path)
        lum = np.array(lum, dtype=np.float32)

        lum_tensor = torch.tensor(lum, dtype=torch.float32).unsqueeze(0)
        return lum_tensor