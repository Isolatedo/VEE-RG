# VEE-RG Architecture Source

This package contains the architecture code for **VEE-RG** (Visit-order
Exchange Equivariance Modeling for Region-Level Longitudinal Report
Generation). It is intentionally source-only: no datasets, checkpoints,
experiment outputs, ablation branches, evaluation scripts, or post-processing
artifacts are included.

## Included modules

- `veerg.model.VEEReportGenerator`: the Stage-2 regional report generator.
- `veerg.visual`: frozen anatomical detector/selector, regional visual adapter,
  detector-valid thresholding, and hard top-8 selection.
- `veerg.longitudinal.paired_region_stabilization.PairedRegionStabilization`:
  query-guided paired-region stabilization with C5 fallback and ROI residuals.
- `veerg.longitudinal.change_factorization.ChangeFactorization`: exchange-
  invariant and visit-order-sensitive change factorization with change and
  direction supervision.
- `veerg.longitudinal.contrastive`: anonymized regional text grouping and
  bidirectional multi-positive InfoNCE.
- `veerg.decoder`: the shared cross-attention PubMed GPT-2 regional decoder.
- `veerg.assembly`: fixed 29-region ordering and report assembly.
- `veerg.detector`: the minimal Faster R-CNN adapter required by the anatomical
  feature interface.

The package does not implement a reversal loss. The exchange behavior is
encoded by the factorized representation itself. No ROI-MLP, ROI-only, C5-off,
EVENT-off, input-off, or other ablation implementation is exposed.

## External inputs

The caller must provide paths to the detector checkpoint, region-selector
checkpoint, frozen longitudinal checkpoint, and PubMed GPT-2 directory. These
files are deliberately not part of this archive. A minimal configuration can
be loaded with `veerg.config.load_config`:

```yaml
data:
  image_size: 512
model:
  detector_checkpoint: /path/to/anatomical_detector.pth
  selector_checkpoint: /path/to/region_selector.pth
  longitudinal_checkpoint: /path/to/vee_rg_longitudinal.pth
  language_model_path: /path/to/PubMed-GPT-2-Medium
  hidden_dim: 1024
  selector_threshold: -1.0
  selector_top_k: 8
```

Use `veerg.model.build_model(config)` after loading the configuration. The
current data loader, training entry points, benchmark scripts, and local path
configuration remain outside this source package by design.

## Tensor contract

The Stage-2 batch passed to `VEEReportGenerator` contains `current_image` and
`prior_image` tensors of shape `[B, 1, 512, 512]`, a boolean
`has_real_prior[B]`, a boolean `region_has_sentence[B, 29]`, and
`region_phrases[B][29]`. The detector interface returns one ROI feature,
validity flag, box, and C5 feature map for each of the 29 anatomical regions.

For Stage 1, `VEEEncoder` consumes the same detector features for
`prior_image` and `current_image`, together with `states[B, 29, 3]`,
`texts[B][29]`, `prior_boxes`, `current_boxes`, `prior_valid`, and
`current_valid` for the auxiliary supervision terms. The detector must be in
evaluation mode when its `forward_spatial` interface is called.
