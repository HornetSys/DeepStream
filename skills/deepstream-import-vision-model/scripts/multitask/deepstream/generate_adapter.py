#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Generate an executable task-type adapter from a routed model manifest.

The implementation is selected from the declared task and tensor contract,
never from the model name. ``implementation_status=implemented`` means the
callable code exists; ``validation_status=generated-untested`` means it is not
runtime-ready. The route remains ADAPTER_REQUIRED until the hooks pass real
DeepStream, semantic-oracle, and rendered-output validation and the readiness
validator changes implementation_status from implemented to tested.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from pprint import pformat
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "model"
SKILL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
from model_pipeline import (  # noqa: E402
    TASK_CONTRACT_REQUIRED_FIELDS,
    ManifestError,
    ModelManifest,
    manifest_from_dict,
)


TASK_RECIPES: dict[str, dict[str, Any]] = {
    "classification": {
        "decoder_contract": "Convert the declared score/logit tensor to top-k class id, score, and label.",
        "renderer_contract": "Overlay top-k labels and scores on the source frame.",
        "oracle": "Expected top-one label and reviewed top-k set.",
    },
    "segmentation": {
        "decoder_contract": "Convert declared logits/mask tensors to an image-aligned class or instance mask.",
        "renderer_contract": "Alpha-blend a deterministic color mask and legend on the source frame.",
        "oracle": "Reviewed mask or ground truth suitable for IoU comparison.",
    },
    "pose": {
        "decoder_contract": "Convert declared heatmaps, offsets, or coordinates to scored keypoints.",
        "renderer_contract": "Draw keypoints and the declared skeleton after coordinate inversion.",
        "oracle": "Reviewed keypoints or ground truth suitable for PCK/OKS comparison.",
    },
    "depth": {
        "decoder_contract": "Convert the declared tensor to aligned depth with scale, units, and invalid-mask policy.",
        "renderer_contract": "Render a deterministic depth colormap with units and valid range.",
        "oracle": "Reviewed depth output or aligned ground truth.",
    },
}


def _classification_reference_supported(manifest: ModelManifest) -> bool:
    """Return whether stock nvinfer can own this classifier's preprocessing."""

    return (
        manifest.task == "classification"
        and (len(manifest.std) <= 1 or len(set(manifest.std)) == 1)
    )

def _required_hooks(manifest: ModelManifest) -> list[str]:
    del manifest
    return [
        "pipeline_runner",
        "pipeline_builder",
        "metadata_probe",
        "output_sink",
        "input_adapter",
        "decoder",
        "renderer",
        "structured_output",
        "semantic_validator",
    ]


def _pipeline_plan(manifest: ModelManifest) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = [
        {"id": "source", "role": "caller-supplied-media-or-tensor-input"},
        {"id": "decode", "role": "media-decode-and-platform-preflight"},
    ]
    nodes.extend(
        {
            "id": stage,
            "role": "required-upstream-stage",
            "status": "must-be-supplied-or-generated-and-tested",
        }
        for stage in manifest.upstream_stages
    )
    nodes.extend(
        [
            {"id": "preprocess", "role": "declared-frame-roi-or-tensor-preprocess"},
            {"id": "model", "role": "nvinfer-or-custom-inference", "source": manifest.source},
            {"id": "adapter", "role": f"{manifest.task}-adapter", "status": "implemented-untested-type-adapter"},
            {"id": "metadata-probe", "role": "task-metadata-and-frame-identity-capture"},
            {"id": "structured-output", "role": "json-evidence"},
            {"id": "semantic-render", "role": "visual-evidence"},
            {"id": "output-sink", "role": "hardware-or-software-encoded-video"},
        ]
    )
    edges: list[dict[str, str]] = []
    previous = ""
    for node in nodes:
        if previous:
            edges.append({"from": previous, "to": node["id"], "contract": "declared task metadata"})
        previous = node["id"]
    return {
        "schema_version": 1,
        "status": "ADAPTER_REQUIRED",
        "task": manifest.task,
        "task_variant": manifest.task_variant,
        "input_domain": manifest.input_domain,
        "nodes": nodes,
        "edges": edges,
        "input_ownership": manifest.pipeline_graph.get("input_ownership") or {
            "owner": "generated type adapter",
            "domain": manifest.input_domain,
        },
        "coordinate_transforms": manifest.pipeline_graph.get("coordinate_transforms") or {
            "policy": "declared task contract; preserve frame/ROI identity",
        },
        "decoder": "adapter.py:decode",
        "renderer": "adapter.py:render",
        "structured_output": "adapter.py:serialize",
        "evidence": {
            "input_fixture": "",
            "backend_parity": "",
            "deepstream_runtime": "",
            "structured_output": "",
            "semantic_validation": "",
            "overlay_evidence": "",
            "rendered_output": "",
            "visual_validation": "",
        },
        "execution_gates": {
            "deepstream": ["startup", "buffer_count_gt_zero", "eos", "task_metadata"],
            "visual": ["decoded_video", "decoded_frame", "structured_task_output", "visible_overlay"],
            "pipeline_visual_result": "PASS only when DeepStream and visual gates both PASS",
        },
        "fallbacks": {
            "headless": True,
            "encoder": ["nvv4l2h264enc", "theoraenc+oggmux", "skipped"],
            "rule": "A fallback may change only the output sink; preserve model, preprocessing, metadata, and renderer contracts.",
        },
    }


def _adapter_source(manifest: ModelManifest, recipe: dict[str, Any]) -> str:
    reference_supported = _classification_reference_supported(manifest)
    spec = {
        "task": manifest.task,
        "task_variant": manifest.task_variant,
        "input_domain": manifest.input_domain,
        "input_name": manifest.input_name,
        "input_shape": list(manifest.input_shape),
        "layout": manifest.layout,
        "color_format": manifest.color_format,
        "scale": manifest.scale,
        "mean": list(manifest.mean),
        "std": list(manifest.std),
        "preprocess_units": manifest.preprocess_units,
        "output_names": list(manifest.output_names),
        "task_contract": manifest.task_contract,
        "upstream_stages": list(manifest.upstream_stages),
        "decoder_contract": recipe["decoder_contract"],
        "renderer_contract": recipe["renderer_contract"],
        "oracle": recipe["oracle"],
        "reference_pipeline_runner": (
            str(Path(__file__).with_name("run_type_adapter_pipeline.py").resolve())
            if reference_supported
            else ""
        ),
        "preprocess_strategy": (
            "nvinfer-native" if reference_supported else "adapter-input-tensor"
        ),
    }
    encoded = pformat(spec, sort_dicts=True, width=100)
    return f'''#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Generated DeepStream task adapter; semantics come only from SPEC."""

from __future__ import annotations

from typing import Any

from type_adapter_runtime import (
    build_pipeline_task,
    decode_task,
    input_adapter_task,
    metadata_probe_task,
    output_sink_task,
    render_task,
    run_pipeline_task,
    serialize_task,
    validate_task,
)

SPEC = {encoded}


def input_adapter(payload: Any, context: dict[str, Any]) -> Any:
    return input_adapter_task(SPEC, payload, context)


def decode(outputs: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    return decode_task(SPEC, outputs, context)


def render(frame: Any, decoded: dict[str, Any], context: dict[str, Any]) -> Any:
    return render_task(SPEC, frame, decoded, context)


def serialize(decoded: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    return serialize_task(SPEC, decoded, context)


def build_pipeline(context: dict[str, Any]) -> Any:
    return build_pipeline_task(SPEC, context)


def metadata_probe(tensor_meta: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    return metadata_probe_task(SPEC, tensor_meta, context)


def output_sink(context: dict[str, Any]) -> dict[str, Any]:
    return output_sink_task(SPEC, context)


def pipeline_runner(context: dict[str, Any]) -> dict[str, Any]:
    return run_pipeline_task(SPEC, context)


def validate(decoded: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    return validate_task(SPEC, decoded, oracle)
'''


def generate(manifest: ModelManifest, output_dir: Path) -> dict[str, Any]:
    if manifest.task == "detection":
        raise ValueError(
            "object detection uses this skill's native parser/KITTI pipeline; "
            "generate_adapter.py is only for classification, segmentation, pose, and depth"
        )
    if manifest.task not in {"classification", "segmentation", "pose", "depth"}:
        raise ValueError(f"no supported type adapter for task {manifest.task!r}")
    if manifest.input_domain != "image":
        raise ValueError(
            "type adapter generation requires input_domain='image'; "
            f"received {manifest.input_domain!r}"
        )
    missing_contract = [
        field
        for field in TASK_CONTRACT_REQUIRED_FIELDS[manifest.task]
        if manifest.task_contract.get(field) in (None, "", [], {})
    ]
    if missing_contract:
        raise ValueError(
            "cannot generate an executable adapter; task_contract is missing: "
            + ", ".join(missing_contract)
        )
    if not manifest.output_names:
        raise ValueError("cannot generate an executable adapter without output_names")
    recipe = TASK_RECIPES[manifest.task]
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / "adapter.py"
    contract_path = output_dir / "adapter-contract.json"
    plan_path = output_dir / "pipeline-plan.json"
    merged_manifest_path = output_dir / "manifest-with-adapter.json"
    runtime_path = output_dir / "type_adapter_runtime.py"
    oracle_validator_path = output_dir / "validate_semantic_oracle.py"
    for path in (
        adapter_path,
        runtime_path,
        oracle_validator_path,
        contract_path,
        plan_path,
        merged_manifest_path,
    ):
        if path.exists():
            raise FileExistsError(
                f"{path} exists; choose a fresh adapter output directory"
            )

    required_hooks = _required_hooks(manifest)
    reference_supported = _classification_reference_supported(manifest)
    contract: dict[str, Any] = {
        "schema_version": 1,
        "task": manifest.task,
        "task_variant": manifest.task_variant,
        "implementation_status": "implemented",
        "validation_status": "generated-untested",
        "required_hooks": required_hooks,
        "input_adapter": "adapter.py:input_adapter",
        "decoder": "adapter.py:decode",
        "renderer": "adapter.py:render",
        "structured_output": "adapter.py:serialize",
        "structured_output_path": "structured-output.json",
        "pipeline_builder": "adapter.py:build_pipeline",
        "pipeline_plan": "pipeline-plan.json",
        "metadata_probe": "adapter.py:metadata_probe",
        "pipeline_runner": "adapter.py:pipeline_runner",
        "reference_pipeline_runner": (
            str(Path(__file__).with_name("run_type_adapter_pipeline.py").resolve())
            if reference_supported
            else ""
        ),
        "output_sink": "adapter.py:output_sink",
        "semantic_validator": "adapter.py:validate",
        "semantic_oracle_schema": "semantic-oracle-v1",
        "semantic_oracle_validator": "validate_semantic_oracle.py",
        "preprocess_strategy": (
            "nvinfer-native" if reference_supported else "adapter-input-tensor"
        ),
        "validation_evidence": {
            "input_fixture": "",
            "backend_parity": "",
            "deepstream_runtime": "",
            "structured_output": "",
            "semantic_validation": "",
            "overlay_evidence": "",
            "rendered_output": "",
            "visual_validation": "",
        },
        "readiness_status": "",
        "validation_required": [
            "reference-to-TensorRT parity on the supplied fixture",
            "one-stream DeepStream EOS and numeric assertion",
            "structured semantic output against the supplied oracle",
            "rendered frame or video with visible task output",
        ],
        "runtime_visual_acceptance": {
            "deepstream_status": "PASS only after startup, nonzero buffers, required metadata, and EOS",
            "visual_status": "PASS only after model-aware structured output, decoded video/frame, visible overlay, and renderer-written overlay evidence bound to the structured-output SHA-256",
            "pipeline_visual_result": "PASS only when deepstream_status and visual_status are both PASS",
            "orthogonal_results": ["backend_status", "semantic_status", "route_status"],
        },
        "root_cause_record": {
            "required_fields": ["gate", "status", "category", "detail", "evidence", "next_action"],
            "rule": "Preserve attempted failures; use BLOCKED for absent resources and UNTESTED for claims without evidence.",
        },
        "implementation_rule": (
            "The type adapter code is implemented but generated-untested. Generate candidate configs with "
            "--allow-implemented-adapter, then run every validation_required gate."
        ),
        "promotion_rule": (
            "Set implementation_status to tested only after every validation_required gate "
            "passes on real supplied input and every validation_evidence field is populated; "
            "then set validation_status=tested."
        ),
    }
    adapter_path.write_text(_adapter_source(manifest, recipe), encoding="utf-8")
    shutil.copy2(Path(__file__).with_name("type_adapter_runtime.py"), runtime_path)
    shutil.copy2(
        Path(__file__).resolve().parents[1]
        / "validation"
        / "validate_semantic_oracle.py",
        oracle_validator_path,
    )
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pipeline_plan = _pipeline_plan(manifest)
    plan_path.write_text(
        json.dumps(pipeline_plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    merged_data = manifest.to_dict()
    # The adapter owns normalized input tensors, including per-channel std.
    # nvinfer consumes tensor meta; no unresolved native preprocess library is
    # claimed by the generated bundle.
    merged_data["nvdspreprocess"] = False
    merged_data["adapter_contract"] = contract
    merged_data["pipeline_graph"] = pipeline_plan
    merged_manifest = manifest_from_dict(merged_data, manifest_path=merged_manifest_path)
    merged_manifest_path.write_text(
        json.dumps(merged_manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "ADAPTER_REQUIRED",
        "output_dir": str(output_dir),
        "generated": [
            str(adapter_path),
            str(runtime_path),
            str(oracle_validator_path),
            str(contract_path),
            str(plan_path),
            str(merged_manifest_path),
        ],
        "next": contract["implementation_rule"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="routed model manifest JSON")
    parser.add_argument("output_dir", type=Path, help="adapter output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output_dir = args.output_dir.resolve()
        if output_dir == SKILL_ROOT or SKILL_ROOT in output_dir.parents:
            raise ValueError("adapter output directory must be outside the installed skill")
        data = json.loads(args.manifest.read_text(encoding="utf-8"))
        manifest = manifest_from_dict(data, manifest_path=args.manifest)
        result = generate(manifest, output_dir)
    except (OSError, ValueError, ManifestError, json.JSONDecodeError) as exc:
        print(
            json.dumps({"status": "BLOCKED", "errors": [str(exc)]}, indent=2),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
