# PRISM / DLPE

Code for **Multi-Modal Monocular Endoscopic Depth and Pose Estimation with Edge-Guided Self-Supervision**.

The canonical DLPE configuration uses RGB + shading/luminance for the depth branch and RGB + edge for the pose branch. This repository contains the depth/pose training code and reference inference code. The IID shading generator and DexiNed/SegCol edge generator are external preprocessing dependencies; their network implementations and checkpoints are not bundled here.

![PRISM structure](_Structure.png)

## What is kept

```text
PRISM/
├── train_prism.py          # supported training entrypoint
├── trainer.py              # stage-2 joint depth/pose trainer
├── options.py              # shared command-line/config options
├── path_config.py          # portable data/weight/output roots
├── datasets/               # endoscopy and KITTI-compatible loaders
├── networks/               # canonical ResNet/Monodepth2-style PRISM model
├── splits/                 # reproducible train/validation/test lists
├── reference/              # inference and evaluation code
│   ├── depth/
│   └── pose_predict_feast_v1.py
└── ablations/              # optional baselines and specialist trainers
    ├── networks/
    │   ├── iid/
    │   ├── monodepth2/
    │   └── monovit/
    └── training/
        ├── trainer_depth.py
        └── trainer_edge.py
```

The old duplicate `train*.py` wrappers are not part of the maintained API. Use `train_prism.py` for all runs. During this cleanup they were moved, without deletion, to `/home/xju/Workspace/PRISM-archive-20260811/legacy_training_entrypoints/`.

## Environment

Create a Python environment with a CUDA-compatible PyTorch build, then install:

```bash
pip install -r requirements.txt
```

The default PRISM path needs PyTorch, torchvision, NumPy, Pillow, OpenCV, SciPy, scikit-image, scikit-learn and tensorboardX. To reproduce the MonoViT ablation and install its additional packages, use:

```bash
pip install -r ablations/requirements.txt
```

Datasets, generated modalities and checkpoints are deliberately not committed. Configure their roots with:

```bash
export PRISM_DATA_ROOT=/path/to/data_folder
export PRISM_WEIGHTS_ROOT=/path/to/weights
export PRISM_OUTPUT_ROOT=/path/to/output_data
```

The more specific variables `PRISM_DATA_PATH`, `PRISM_GENERATED_PATH`, `PRISM_WEIGHTS_PATH`, and `PRISM_OUTPUT_PATH` override individual locations.

## Training code

PRISM uses the following three-stage workflow:

1. Pre-generate shading/luminance and edge maps with the external IID and edge models.
2. Jointly train depth and pose with `trainer.py`.
3. Load the joint checkpoint, freeze the depth encoder/decoder, and fine-tune pose with the specialist trainer in `ablations/training/`.

### 1. Prepare modalities

For an RGB image such as `sequence/00001.jpg`, the loaders expect generated modalities under the configured generated-data root. Representative layouts are:

```text
generated/
├── hk/
│   ├── edge/sequence/avg/00001.png.npy
│   └── shading/sequence/decomposed/light00001.png
└── c3vd/
    ├── edge/sequence/avg/0000_color.png.npy
    └── shading/sequence/decomposed/light0000_color.png
```

Check the selected split and `datasets/HK_dataset.py` before launching a new dataset because naming conventions differ between HK, C3VD and EndoMapper exports.

### 2. Joint depth/pose training

```bash
python train_prism.py --pipeline joint \
  --model_name hk_mono_finetuned_dlpe_edge_ssim \
  --dataset hk --data hk --split hk \
  --height 288 --width 288 \
  --training_mode both --edge_loss
```

The modality convention encoded by the trainers is:

| Model-name token | Depth input | Pose input |
| --- | --- | --- |
| `dlpe` | RGB + luminance | RGB + edge |
| `depl` | RGB + edge | RGB + luminance |
| `both_edge` | RGB + edge | RGB + edge |
| `both_lum` | RGB + luminance | RGB + luminance |
| `depth_edge` / `depth_lum` | selected extra depth channel | RGB |
| `pose_edge` / `pose_lum` | RGB | selected extra pose channel |

These strings are functional configuration, not merely experiment labels; do not rename a checkpoint without preserving the corresponding modality choice.

### 3. Freeze depth and fine-tune pose

Use `edge_pose_resume` to load the joint checkpoint and invoke the frozen-depth pose trainer:

```bash
python train_prism.py --pipeline edge_pose_resume \
  --model_name hk_mono_finetuned_dlpe_edge_ssim_depth_fix_thorough_aug \
  --dataset hk --data hk --split hk \
  --height 288 --width 288 \
  --training_mode pose --edge_loss
```

`edge_pose_scratch` initializes the pose branch from the generic Monodepth2 pose checkpoint instead. The legacy `depth` and `edge` pipeline aliases remain available for reproducing older experiments, but both specialist trainers freeze depth and optimize pose; they are not depth-only trainers despite the historical filename `trainer_depth.py`.

Options may also be stored in JSON and passed with `--config path/to/config.json`. Command-line values override the corresponding defaults according to `options.py`.

## Reference code

`predict_prism.py` is the supported dispatcher. The implementation modules live under `reference/` so they are separated from training code.

Depth inference/evaluation for C3VD-style folders:

```bash
python predict_prism.py depth-c3vd \
  --image_path /path/to/c3vd \
  --model_basepath /path/to/weights \
  --model_name experiment/models/weights_19 \
  --output_path /path/to/output \
  --edge_root /path/to/generated/edge \
  --shading_root /path/to/generated/shading \
  --eval
```

Other maintained dispatch commands are:

```bash
python predict_prism.py depth-endomapper [arguments...]
python predict_prism.py depth-endomapper-288 [arguments...]
python predict_prism.py pose-c3vd [arguments...]
```

Run `python predict_prism.py COMMAND --help` to see the arguments implemented by a reference module. Checkpoints are expected to contain the usual `encoder.pth`, `depth.pth`, `pose_encoder.pth`, and `pose.pth` files as required by the selected command.

## Ablations

`ablations/` is intentionally importable but is not the default code path:

- `ablations/networks/iid`: historical IID-compatible encoder/decoders used by reference comparisons.
- `ablations/networks/monodepth2`: archived plain Monodepth2 baseline.
- `ablations/networks/monovit`: optional MonoViT backbone.
- `ablations/training`: frozen-depth pose fine-tuning implementations used by stage 3 and older edge-loss experiments.

Track-point, five-channel, optical-flow, C3VD fine-tuning and CLiMB submission experiments are not mixed into this original repository. They remain in the separate `PRISM-CLiMB` workspace, which prevents challenge-specific code from silently changing the reference DLPE implementation.

## Reproducibility notes

- Image height and width must be multiples of 32.
- A checkpoint's behavior depends on both its tensors and the model-name modality token.
- Preserve the exact preprocessing checkpoint and generated edge/shading files used for an experiment.
- Split files may contain absolute paths from the original machine; inspect them before reuse.
- Outputs, weights, datasets, caches and logs are ignored by Git.

![Qualitative results](Main_qualitative_031026_w_shading.png)
