<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Semantic oracle contract

Read this reference before claiming semantic, runtime, visual, or overall
`PASS`. An oracle is category-specific expected output for fixed real-media
fixtures; model output, a visually plausible overlay, or an unreviewed label is
not an oracle.

## Required document

Supply one JSON document with:

- `schema_version: 1`;
- one supported `task`;
- `oracle_type`: `ground-truth` or `reviewed`;
- `provenance.source`; a reviewed oracle also requires `reviewer` and a
  timezone-bearing `reviewed_at`;
- `precision`;
- `acceptance.metric`, `comparator`, `threshold`, and an explicit positive
  `minimum_samples`;
- at least `minimum_samples` entries with unique `fixture_id`, `media.path`,
  optional frame/ROI identity, and category-specific `expected` output.

The validator accepts both canonical routed names
(`object-detection`, `image-classification`, `semantic-segmentation`,
`2d-pose`, `monocular-depth`) and their short oracle names
(`detection`, `classification`, `segmentation`, `pose`, `depth`).

```json
{
  "schema_version": 1,
  "task": "segmentation",
  "oracle_type": "reviewed",
  "provenance": {
    "source": "reviewed fixture set v1",
    "reviewer": "tester@example.com",
    "reviewed_at": "2026-09-17T10:00:00-07:00"
  },
  "precision": "fp16",
  "acceptance": {
    "metric": "mean_iou",
    "comparator": "gte",
    "threshold": 0.99,
    "minimum_samples": 1
  },
  "samples": [
    {
      "fixture_id": "street-frame-0001",
      "media": {"path": "street.mp4", "frame_index": 1},
      "expected": {"mask_path": "street-frame-0001-mask.npy"}
    }
  ]
}
```

Paths in `expected.mask_path` or `expected.depth_path` must be relative regular
files beneath the oracle document directory. The runtime harness resolves an
artifact to an in-memory `mask` or `depth` array before calling the generated
adapter's `validate` hook.

## Category contract

| Task | Expected field | Metric and default gate |
|---|---|---|
| Detection | `boxes` with reviewed class and geometry | `map`, `gte`; use model/test-plan threshold and reviewed IoU/precision/recall evidence |
| Classification | integer `class_id`; optional reviewed `top_k` | `top1_accuracy >= 1.0` |
| Segmentation | inline `mask` or `mask_path` | `mean_iou >= 0.99`; `>= 0.98` for INT8 |
| 2D pose | ordered `keypoints` matching the declared skeleton | `pck_oks_agreement >= 0.99` |
| Monocular depth | inline `depth` or `depth_path`, with declared units/range | `direct_rmse_fraction <= 0.01` |

A stricter model-owned threshold is allowed. Never relax these defaults to
turn a failure into a pass. `minimum_samples` is supplied by the test plan and
must be at least one; every selected sample must pass. Missing or invalid
oracle data is `BLOCKED`/`UNTESTED`, not `FAIL` and never `PASS`.

## Mandatory preflight

```bash
python3 scripts/multitask/validation/validate_semantic_oracle.py \
  oracle.json --task semantic-segmentation
```

Exit `0` means the contract is valid. Exit `2` means `BLOCKED`; report every
returned error and stop semantic and visual promotion. After preflight, bind
each result to its `fixture_id`, frame/ROI identity, structured-output hash,
oracle hash, observed metric, threshold, and status.
