<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Model and media selection for a 13-case application campaign plus one classifier

This is a source map, **not** evidence that any model ran. First inspect the
model card and its versioned file browser, download the exact model file plus
its labels/config/parser contract, and record the selected version and local
SHA-256 in the caller's case manifest. Do not silently replace a campaign's
pinned ONNX with a similarly named, different architecture or revision. The
expected vendor checksum, if supplied, belongs in the test case; the skill
records the observed file hash to bind its own evidence.

The [DeepStream SDK sample-stream inventory](https://docs.nvidia.com/metropolis/deepstream/8.0/text/DS_sample_configs_streams.html)
provides videos/images under
`/opt/nvidia/deepstream/deepstream/samples/streams/`. NGC model packages
generally provide model/config/labels, not the campaign video or a semantic
ground-truth mask. Use a caller-supplied representative clip if the SDK sample
does not contain the target objects. Inspect decoded frames before treating a
video as a meaningful fixture. Preserve the original detector benchmark's
`sample_720p.mp4` default; other selections are explicit case inputs, not a
silent change to that default.

In the existing input bundle, sampled frames at 0, 10, 35, and 50 seconds
of `person-bicycle-car-detection.mp4`, and at 0, 10, 20, and 28 seconds of
`car-detection.mp4`, showed empty parking pavement rather than the expected
traffic/vehicle ROIs. This does not prove every frame is empty, but those
clips are not established visual fixtures for the named cases. By contrast,
SDK `sample_720p.mp4` and `sample_1080p_h264.mp4` at 10 seconds visibly show
cars and people. Choose a fresh case fixture, sample multiple timestamps,
record the selected file hash and frame/time oracle, and regenerate reports;
do not relabel prior blocked PDFs as passes.

| Application case | Pinned model source to inspect | Suitable input and required stage |
|---|---|---|
| TrafficCamNet | [NGC `pruned_onnx_v1.0.4`](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/trafficcamnet/pruned_onnx_v1.0.4/file-browser): `resnet18_trafficcamnet_pruned.onnx` plus labels/config | SDK `sample_720p.mp4` for the legacy benchmark; use an explicitly selected traffic clip for traffic-specific visual checks. |
| PeopleNet | [NGC `deployable_quantized_onnx_v2.6.3`](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/peoplenet/deployable_quantized_onnx_v2.6.3/file-browser): `resnet34_peoplenet.onnx` plus labels/config | SDK `sample_720p.mp4` only if people are visible; otherwise a caller-supplied people clip. The similarly named INT8 ONNX in an older bundle must be matched to its own revision. |
| YOLOv8n | [Hugging Face `webml/yolov8n` revision `85bc8d7ab30d4065a41909d756c42819b67b4388`](https://huggingface.co/webml/yolov8n/blob/85bc8d7ab30d4065a41909d756c42819b67b4388/onnx/yolov8n.onnx): `onnx/yolov8n.onnx`, observed SHA-256 `190ba5f1e61411a001683e349d6b2cdb0804c0dc67a5e34cd8ff6fd00ee54b4d`; COCO labels from [Ultralytics v8.3.0](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/cfg/datasets/coco.yaml) | SDK `sample_720p.mp4` has visible people and cars. Inspect the exact output contract `[1,84,8400]` and only then use the checked `scripts/deepstream/parsers/yolo_dense_xywh.cpp` dense-XYWH parser, `cluster-mode=2`, class count 80, and a frame-aligned detection oracle. This Hugging Face repository is not an NGC package. |
| RT-DETR | Resolve the selected [Hugging Face RT-DETR](https://huggingface.co/PekingU/rtdetr_r50vd) revision or retain the campaign's pinned `rtdetr_r18vd_heads.onnx` | SDK `sample_720p.mp4` or a traffic clip. The R50 and R18 models are **not** interchangeable. |
| LPD-US | [NGC LPDNet](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/lpdnet/-) and its exact US deployable version | Caller-supplied high-resolution traffic video with readable US license plates; SDK `sample_1080p_h264.mp4` alone is not a plate-visibility oracle. Distinguish plate detection from LPR text recognition. |
| VehicleMakeNet | [NGC `pruned_onnx_v1.1.0`](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/vehiclemakenet/pruned_onnx_v1.1.0/file-browser): ONNX, labels, config | SDK `sample_1080p_h264.mp4` or a car clip; run a vehicle detector and pass its real vehicle ROIs to the secondary classifier. |
| VehicleTypeNet | [NGC `pruned_onnx_v1.1.0`](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/vehicletypenet/pruned_onnx_v1.1.0/file-browser): ONNX, labels, config | Same detector-owned vehicle ROI cascade; do not classify the full frame as a car crop. See [NVIDIA's detector-to-secondary example](https://docs.nvidia.com/deepstream/9.2/text/DS_plugin_gst-nvdspostprocess). |
| CitySemSegFormer | [NGC `deployable_onnx_v1.0`](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/citysemsegformer/-/file-browser): `citysemsegformer.onnx`, `labels.txt`, `nvinfer_config.txt` | SDK `sample_720p.mp4` for the [documented `ds-tao-segmentation` example](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/citysemsegformer/-). Capture the mask rendered on the frame and compare to a reviewed frame-aligned mask before semantic PASS. |
| SegFormer | Retain the exact campaign `segformer_b0_ade.onnx` source/revision; a CitySemSegFormer checkpoint is not a substitute | Caller-supplied scene clip or SDK `sample_office.mp4` if its classes are relevant; require ADE class map and mask geometry. |
| BodyPose2D | Retain the exact campaign checkpoint; [NGC BodyPoseNet](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/bodyposenet/deployable_onnx_v1.0.1) is a different candidate until tensor/decoder compatibility is proved | SDK `yoga.mp4` or `sample_run.mov` if people are visible. Run person detection/ROI extraction when required by the selected pose model; validate keypoints and skeleton overlay. |
| Depth Anything V2 | Retain the campaign's exact `depth_anything_v2_vits.onnx`; [NGC NvDepthAnythingV2 large](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/nvdepthanythingv2/deployable_relative_depthanythingv2_large_v1.0) is a **different** artifact | SDK `sample_run.mov` or another real scene clip; declare relative versus metric depth and the render range. A heatmap alone is not numeric accuracy evidence. |
| FoundationStereo | [NGC fixed-shape small model](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/foundationstereo/deployable_foundation_stereo_small_v1.0/file-browser): select the actual 320x736 or 576x960 ONNX | Synchronized, rectified left/right frames from a real calibrated stereo capture, with calibration and disparity/validity oracle. The [Middlebury Adirondack pair](https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Adirondack-imperfect/) supplies `im0.png`, `im1.png`, `calib.txt`, and `disp0.pfm`; preserve identical crops and update calibration/validity geometry. A shifted copy of one video is not a stereo fixture. |
| Bi3D | [NGC Bi3D v2.0.0](https://catalog.ngc.nvidia.com/orgs/nvidia/isaac/models/bi3d_proximity_segmentation/-): both FeatNet and SegNet | One complete two-stage application on the same [official paired stereo fixture](https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Adirondack-imperfect/) and configured disparity planes; neither subnetwork alone is a visual application PASS. The FeatNet→cost-volume→SegNet adapter must be implemented before runtime certification. |
| MobileNetV2 ImageNet (optional 14th case) | [Hugging Face `onnx-community/mobilenet_v2_0.35_96` pinned revision `f37d9b91354e5a43dae384840f46119c67a76d58`](https://huggingface.co/onnx-community/mobilenet_v2_0.35_96/tree/f37d9b91354e5a43dae384840f46119c67a76d58): `onnx/model.onnx`, `config.json`, `preprocessor_config.json`; observed ONNX SHA-256 `21e68c215414392406550be5d209aac8a4aa6c0b5611b60125b888ca8bda957b` | RGB image/video; resize shortest edge to 128, center-crop 96×96, then normalize `(pixel/255-0.5)/0.5`. A synthetic golden-retriever image looped to video is a diagnostic fixture only; use an independently reviewed image label for semantic validation and real media for field accuracy. The reference runner accepts `--mux-width`, `--mux-height`, and one explicit `--encoder theora` retry if hardware H.264 initialization fails. |

For each case, keep `source`, `components`, `inputs` with roles, the model's
labels/config, preprocess/postprocess contract, required upstream stages, and
the exact media/oracle provenance together. Run diagnostic backend and
DeepStream attempts with the available model/media even when a reviewed
accuracy oracle is absent; record semantic `UNTESTED` and do not promote to a
functional PASS. Stereo without a real pair/calibration, missing model files,
or absent required components remain blocked before an application run.
