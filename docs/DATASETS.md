# Dataset configurations and splits

The machine-readable version of this document is
[`configs/datasets.json`](../configs/datasets.json). The committed split files
are the source of truth; counts below exclude no additional samples at runtime.

## Hyper-Kvasir

PRISM uses selected lower-GI video frames with BBPS 2–3. The canonical `hk`
split contains 16,976 training frames and 1,887 validation frames from 29
unique video sequences. Sequence UUIDs are listed in
`configs/datasets.json`.

The historical training split is frame-based rather than fully
sequence-exclusive:

- `46e2df1d-9fb9-4fc0-b09e-0b90899350d7` occurs in train and validation;
- its frames 1–454 are used for training and frames 455–614 for validation;
- no RGB frame occurs in both files.

Validation is used by the training loop for loss/metric monitoring. Hyper-Kvasir
is not used as a held-out quantitative test dataset in PRISM, so unused HK test
files are intentionally not included. Quantitative evaluation uses C3VD, while
EndoMapper is used qualitatively.

The released `hk` split contains 16,976 training frames and 1,887 validation
frames. Historical interval-sampling variants belong to the ablation study and
are archived separately under `ablations/splits/`; they are not PRISM defaults.

## C3VD

The canonical `c3vd_mysplit` is sequence-exclusive and contains all 22 C3VD
sequences used by this repository.

- Train (6,819 frames): `cecum_t1_a`, `cecum_t1_b`, `cecum_t2_a`,
  `cecum_t2_b`, `cecum_t2_c`, `cecum_t3_a`, `sigmoid_t1_a`, `sigmoid_t2_a`,
  `trans_t1_a`, `trans_t1_b`, `trans_t2_a`, `trans_t2_b`, `trans_t2_c`,
  `trans_t3_a`, `trans_t3_b`.
- Validation (1,454 frames): `cecum_t4_a`, `sigmoid_t3_a`, `trans_t4_a`.
- Test (1,698 frames): `cecum_t4_b`, `desc_t4_a`, `sigmoid_t3_b`,
  `trans_t4_b`.

The released `c3vd_mysplit` contains 6,819 training frames, 1,454 validation
frames, and 1,698 test frames. Historical interval-sampling variants are kept
only under `ablations/splits/`.

## EndoMapper

EndoMapper is used for qualitative cross-domain evaluation, not for the
released original DLPE training split. Consequently, this repository does not
commit an EndoMapper train/validation split. Prepare evaluation frames using
the same RGB, edge, and luminance numerical contract described in
[`DATA_PREPARATION.md`](DATA_PREPARATION.md).

## Portable paths

Every committed line uses the portable form `<sequence>/<filename>`. The RGB
path is resolved relative to `--data_path`; generated modalities are resolved
under `PRISM_GENERATED_PATH`. Absolute paths are still accepted for private
experiments, but must not be committed.
