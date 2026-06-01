from __future__ import absolute_import, division, print_function

import numpy as np


# Stereo-supervised Monodepth models are trained with a nominal 0.1-unit
# baseline. KITTI's stereo rig baseline is 54 cm, so depths are scaled by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
