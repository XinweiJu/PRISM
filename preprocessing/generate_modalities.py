#!/usr/bin/env python3
"""Generate the edge and luminance inputs consumed by PRISM's HKDataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state(path: Path, device: torch.device):
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise TypeError("Expected a state dictionary in {}".format(path))
    return state


class ShadingGenerator:
    """IID/SHADES luminance generator with the original PRISM convention."""

    def __init__(self, checkpoint_dir: Path, device: torch.device, size: int):
        sys.path.insert(0, str(REPO_ROOT))
        from ablations.networks.iid.decompose_decoder import decompose_decoder
        from networks import ResnetEncoder

        self.device = device
        self.size = size
        self.encoder = ResnetEncoder(18, False)
        self.decoder = decompose_decoder(self.encoder.num_ch_enc, scales=range(4))
        encoder_state = load_state(checkpoint_dir / "decompose_encoder.pth", device)
        encoder_state = {
            key: value
            for key, value in encoder_state.items()
            if key in self.encoder.state_dict()
        }
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.decoder.load_state_dict(
            load_state(checkpoint_dir / "decompose.pth", device), strict=True
        )
        self.encoder.to(device).eval()
        self.decoder.to(device).eval()

    @torch.inference_mode()
    def __call__(self, image_rgb: Image.Image) -> np.ndarray:
        original_hw = (image_rgb.height, image_rgb.width)
        resampling = getattr(Image, "Resampling", Image)
        resized = image_rgb.resize((self.size, self.size), resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(array))[None].to(self.device)
        _, light = self.decoder(self.encoder(tensor))
        light = F.interpolate(light, size=original_hw, mode="bilinear", align_corners=False)
        return light[0, 0].mul(255).clamp(0, 255).byte().cpu().numpy()


class EdgeGenerator:
    """DexiNed average-side edge generator matching the PRISM preprocessing."""

    def __init__(
        self, dexined_repo: Path, checkpoint: Path, device: torch.device, size: int
    ):
        sys.path.insert(0, str(dexined_repo.resolve()))
        from model import DexiNed

        self.device = device
        self.size = size
        self.model = DexiNed().to(device).eval()
        self.model.load_state_dict(load_state(checkpoint, device), strict=True)

    @torch.inference_mode()
    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        resized = cv2.resize(
            image_bgr, (self.size, self.size), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        tensor = torch.from_numpy(
            np.ascontiguousarray((resized - MEAN_BGR).transpose(2, 0, 1))
        )[None].to(self.device)
        sides = []
        for output in self.model(tensor):
            side = torch.sigmoid(output)[0, 0].float().cpu().numpy()
            low, high = float(side.min()), float(side.max())
            side = ((side - low) * 255.0 / max(high - low, 1e-8)).astype(np.uint8)
            sides.append(
                cv2.resize(side, (width, height), interpolation=cv2.INTER_CUBIC)
            )
        return np.mean(np.asarray(sides, dtype=np.float32), axis=0) / 255.0


def image_paths(root: Path, dataset: str) -> list[Path]:
    pattern = "*_color.png" if dataset == "c3vd" else "*.jpg"
    paths = sorted(path for path in root.glob("*/*") if path.match(pattern))
    if not paths:
        raise FileNotFoundError("No {} images found under {}".format(pattern, root))
    return paths


def output_paths(image: Path, input_root: Path, output_root: Path, dataset: str):
    relative = image.relative_to(input_root)
    sequence = relative.parent
    if dataset == "c3vd":
        edge_name = image.name + ".npy"
        lum_name = "light" + image.name
    else:
        edge_name = image.stem + ".png.npy"
        lum_name = "light" + image.stem + ".png"
    return (
        output_root / dataset / "edge" / sequence / "avg" / edge_name,
        output_root / dataset / "shading" / sequence / "decomposed" / lum_name,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["hk", "c3vd"], required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--iid-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dexined-repo", type=Path, required=True)
    parser.add_argument("--dexined-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iid-size", type=int, default=288)
    parser.add_argument("--edge-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, help="process only N images for a smoke test")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    paths = image_paths(args.input_root, args.dataset)
    if args.limit is not None:
        paths = paths[: args.limit]
    edge_generator = EdgeGenerator(
        args.dexined_repo, args.dexined_checkpoint, device, args.edge_size
    )
    shading_generator = ShadingGenerator(
        args.iid_checkpoint_dir, device, args.iid_size
    )

    written = 0
    skipped = 0
    for index, image_path in enumerate(paths, 1):
        edge_path, lum_path = output_paths(
            image_path, args.input_root, args.output_root, args.dataset
        )
        if not args.overwrite and edge_path.exists() and lum_path.exists():
            skipped += 1
            continue
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError("Could not read {}".format(image_path))
        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        edge_path.parent.mkdir(parents=True, exist_ok=True)
        lum_path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite or not edge_path.exists():
            np.save(edge_path, edge_generator(bgr).astype(np.float32))
        if args.overwrite or not lum_path.exists():
            if not cv2.imwrite(str(lum_path), shading_generator(rgb)):
                raise OSError("Could not write {}".format(lum_path))
        written += 1
        if index % 100 == 0:
            print("processed {}/{}".format(index, len(paths)), flush=True)

    manifest = {
        "dataset": args.dataset,
        "input_root": str(args.input_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "images_found": len(paths),
        "images_written": written,
        "images_skipped": skipped,
        "edge": {
            "network_size": [args.edge_size, args.edge_size],
            "checkpoint": str(args.dexined_checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.dexined_checkpoint),
            "format": "float32 .npy, seven sigmoid sides independently min-max normalized then averaged, range [0,1]",
        },
        "luminance": {
            "network_size": [args.iid_size, args.iid_size],
            "checkpoint_dir": str(args.iid_checkpoint_dir.resolve()),
            "encoder_sha256": sha256(args.iid_checkpoint_dir / "decompose_encoder.pth"),
            "decoder_sha256": sha256(args.iid_checkpoint_dir / "decompose.pth"),
            "format": "uint8 grayscale PNG, sigmoid illumination multiplied by 255, range [0,255]",
        },
    }
    manifest_path = args.output_root / args.dataset / "generation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
