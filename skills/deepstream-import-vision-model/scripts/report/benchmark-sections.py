#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restore Step 8 benchmark sections from verified logs; never change gate results."""

from __future__ import annotations

import configparser
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("source_perf", HERE.parent / "deepstream/parse-perf.py")
assert spec is not None and spec.loader is not None
perf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perf)
TITLES = {5: "trtexec Results", 6: "PEAK_GPU_STREAMS Derivation",
          8: "DeepStream Benchmark Results", 9: "trtexec vs DeepStream Comparison",
          10: "Efficiency Analysis"}


def verified_file(root: Path, record: dict) -> Path:
    path = Path(record["path"])
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("missing benchmark file")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if record.get("sha256") != digest:
        raise ValueError("benchmark file digest differs")
    return path


def table(headers: list, rows: list) -> str:
    def cell(value):
        if isinstance(value, float):
            value = f"{value:.2f}"
        return str(value).replace("|", "&#124;").replace("\n", " ").replace("<", "&lt;").replace(">", "&gt;").replace("_", "_\u200b")
    return "\n".join("| " + " | ".join(map(cell, row)) + " |"
                     for row in [headers, ["---"] * len(headers), *rows])


def collect(manifest: dict, root: Path) -> dict | None:
    record = manifest["benchmark"]
    engine = verified_file(root, record["engine"])
    unit = record["input_unit"]
    if unit not in {"frame", "roi"}:
        raise ValueError("unknown TensorRT input unit")
    trt = {}
    for key in ("bs1", "bsmax"):
        item = record["trtexec"][key]
        batch = item["batch"]
        if type(batch) is not int or batch < 1 or (key == "bs1" and batch != 1):
            raise ValueError("invalid TensorRT batch")
        text = verified_file(root, item["log"]).read_text()
        if ("&&&& PASSED TensorRT.trtexec" not in text or "Data transfers: Disabled" not in text
                or f"--loadEngine={engine}" not in text):
            raise ValueError("TensorRT log is not a successful matching GPU-only benchmark")
        shapes = re.findall(r"Input inference shape\s*:\s*.*?=(\d+)x", text)
        if not shapes:
            shapes = re.findall(r"Input binding for .*? with dimensions (\d+)x", text)
        if not shapes or any(int(value) != batch for value in shapes):
            raise ValueError("TensorRT log batch differs")
        def number(pattern, text=text):
            matches = re.findall(pattern, text)
            value = float(matches[-1]) if matches else float("nan")
            if not math.isfinite(value) or value <= 0:
                raise ValueError("missing or invalid TensorRT measurement")
            return value
        qps = number(r"Throughput:\s*([\d.eE+-]+)\s*qps")
        trt[key] = {"batch": batch, "qps": qps, "imgs_per_sec": qps * batch,
                    "gpu_mean_ms": number(r"GPU Compute Time:.*?mean = ([\d.eE+-]+)"),
                    "p99_ms": number(r"GPU Compute Time:.*?percentile\(99%\) = ([\d.eE+-]+)")}
    if trt["bsmax"]["batch"] < trt["bs1"]["batch"]:
        raise ValueError("invalid maximum batch")
    runs = {}
    if record.get('runner') == 'task-capacity':
        for key in ('run1', 'run2'):
            item = record['deepstream'][key]
            result = json.loads(verified_file(root, item['result']).read_text())
            text = verified_file(root, item['log']).read_text()
            if result.get('producer') != 'deepstream-task-capacity-runner' or result.get('errors') or not result.get('startup'):
                raise ValueError('task capacity did not execute a valid complete graph')
            if result.get('task') != manifest.get('task') or result.get('campaign_id') != manifest.get('campaign_id'):
                raise ValueError('task capacity identity differs')
            if result.get('scope') != 'decode-mux-complete-inference-cascade-task-postprocessing-fakesink':
                raise ValueError('unknown task capacity scope')
            if result.get('rendering_included') is not False or result.get('encoding_included') is not False:
                raise ValueError('task capacity graph scope differs')
            streams = result['streams']
            if type(streams) is not int or not 1 <= streams <= 512:
                raise ValueError('invalid source count')
            if item.get('streams') != streams:
                raise ValueError('declared task capacity source count differs')
            if streams == 1:
                return None  # A single stream cannot establish multi-stream capacity.
            windows = [json.loads(line.split('TASK_CAPACITY ', 1)[1]) for line in text.splitlines() if line.startswith('TASK_CAPACITY ')]
            if windows != result.get('windows') or len(windows) < 5:
                raise ValueError('task capacity windows differ from executed log')
            if result.get('duration_seconds', 0) < 60:
                raise ValueError('task capacity interval is too short')
            per_source = []
            for source_id in range(streams):
                values = []
                for window in windows[-5:]:
                    if len(window['fps']) != streams or window['duration_seconds'] < 2:
                        raise ValueError('incomplete task capacity source/window')
                    value = window['fps'][source_id]
                    if not math.isfinite(value) or value < 0:
                        raise ValueError('invalid task capacity FPS')
                    if len(window.get('task_records', [])) != streams or not all(type(v) is int and v > 0 for v in window['task_records']):
                        raise ValueError('task producer not established in each final source/window')
                    values.append(value)
                per_source.append({'minimum_fps': min(values), 'average_fps': sum(values)/len(values)})
            for counts in ('frames_per_source', 'task_records_per_source'):
                if len(result.get(counts, [])) != streams or not all(type(v) is int and v > 0 for v in result[counts]):
                    raise ValueError('missing per-source execution/task evidence')
            stages = result.get('stages', [])
            if not stages:
                raise ValueError('missing task graph')
            for stage in stages:
                path = verified_file(root, {'path': stage['config'], 'sha256': stage['config_sha256']})
                selected_engine = verified_file(root, {'path': stage['engine'], 'sha256': stage['engine_sha256']})
                parser = configparser.ConfigParser(interpolation=None)
                parser.read(path)
                selected = Path(parser.get('property', 'model-engine-file'))
                selected = selected if selected.is_absolute() else path.parent / selected
                if selected.resolve() != selected_engine or parser.getint('property', 'batch-size') != stage['inference_batch']:
                    raise ValueError('task stage engine/batch differs')
            if stages[-1]['engine_sha256'] != record['engine']['sha256'] or stages[-1]['inference_batch'] > trt['bsmax']['batch']:
                raise ValueError('task benchmark final engine/profile differs')
            if parser.get('property', 'network-type') == '100' and not result.get('decoder'):
                raise ValueError('tensor capacity decoder provenance is missing')
            if result.get('decoder'):
                verified_file(root, result['decoder'])
            aggregate = sum(v['average_fps'] for v in per_source)
            runs[key] = {'streams': streams, 'mux_batch': result['mux_batch'], 'stages': stages,
                         'per_stream': per_source, 'total_fps': aggregate,
                         'fps_per_stream': aggregate/streams, 'worst_fps': min(v['minimum_fps'] for v in per_source),
                         'status': 'PASS' if all(v['minimum_fps'] >= 30 for v in per_source) else 'FAIL'}
        return {'model_name': manifest['case_id'], 'engine': str(engine), 'max_bs': trt['bsmax']['batch'],
                'trtexec': trt, 'deepstream': runs, 'scope': 'task-application-fakesink', 'input_unit': unit,
                'runner': 'task-capacity', 'peak_gpu_streams': math.floor(trt['bsmax']['imgs_per_sec']/30)}
    scope = record.get("scope")
    if scope not in {"decode-mux-infer-parser-fakesink", "task-application-fakesink"}:
        raise ValueError("benchmark graph scope is missing or unsupported")
    for key in ("run1", "run2"):
        item = record["deepstream"][key]
        streams = item["streams"]
        if (type(streams) is not int or streams < 1 or type(item.get("batch")) is not int
                or item["batch"] != streams or type(item.get("exit_code")) is not int or item["exit_code"] != 0):
            raise ValueError("DeepStream needs successful batch=streams execution")
        if streams > trt["bsmax"]["batch"]:
            raise ValueError("DeepStream batch exceeds the benchmarked engine profile")
        config_path = verified_file(root, item["config"])
        config = configparser.ConfigParser(interpolation=None)
        config.read_string(config_path.read_text())
        selected = Path(config.get("property", "model-engine-file"))
        if not selected.is_absolute():
            selected = config_path.parent / selected
        if selected.resolve() != engine or config.getint("property", "batch-size") != streams:
            raise ValueError("DeepStream config engine/batch differs")
        measurement = perf.parse(verified_file(root, item["log"]).read_text(), streams)
        if measurement["window_count"] < 5:
            raise ValueError("insufficient DeepStream measurement windows")
        measurement["fps_per_stream"] = measurement["aggregate_average_fps"] / streams
        measurement["total_fps"] = measurement["aggregate_average_fps"]
        measurement["worst_fps"] = min(s["minimum_fps"] for s in measurement["per_stream"])
        runs[key] = measurement
    return {"model_name": manifest["case_id"], "engine": str(engine), "max_bs": trt["bsmax"]["batch"],
            "trtexec": trt, "deepstream": runs, "scope": scope, "input_unit": unit,
            "peak_gpu_streams": math.floor(trt["bsmax"]["imgs_per_sec"] / 30)}


def sections(data: dict | None, reason: str = "No verified two-run batched benchmark is bound to this case.") -> dict:
    if data is None:
        result = {n: "UNTESTED: " + reason for n in TITLES}
        result[9] += "\n\n" + table(["Metric", "trtexec MAX_BS", "DS Run 1", "DS Run 2"],
                                    [["Comparable throughput", "UNTESTED", "UNTESTED", "UNTESTED"]])
        return result
    trt, runs = data["trtexec"], data["deepstream"]
    comparable = data["input_unit"] == "frame"
    efficiency = lambda run: f'{run["total_fps"] / trt["bsmax"]["imgs_per_sec"] * 100:.2f}%' if comparable else "N/A: ROI/s differs from frames/s"
    result = {5: table(["Batch", "QPS", "Input samples/s", "GPU mean ms", "GPU P99 ms"],
                       [[r["batch"], r["qps"], r["imgs_per_sec"], r["gpu_mean_ms"], r["p99_ms"]] for r in trt.values()]),
              6: (f'Engine-only sizing estimate: floor({trt["bsmax"]["imgs_per_sec"]:.2f} / 30) = {data["peak_gpu_streams"]}. '
                  "Not a measured camera ceiling." if comparable else "N/A: standalone ROI throughput cannot establish camera capacity."),
              8: table(["Run", "Streams = batch", "Avg FPS/source", "Worst sustained FPS/source", "Aggregate frames/s", "≥30 FPS every source"],
                       [[label, r["streams"], r["fps_per_stream"], r["worst_fps"], r["total_fps"], r["status"]]
                        for label, r in zip(["Calibration", "Validation"], runs.values())]),
              9: table(["Metric", "trtexec MAX_BS", "DS Run 1", "DS Run 2"], [
                  ["Engine", Path(data["engine"]).name, Path(data["engine"]).name, Path(data["engine"]).name],
                  ["Batch / streams", data["max_bs"], runs["run1"]["streams"], runs["run2"]["streams"]],
                  ["Throughput", f'{trt["bsmax"]["imgs_per_sec"]:.2f} {data["input_unit"]} samples/s', runs["run1"]["total_fps"], runs["run2"]["total_fps"]],
                  ["Avg FPS/source", "Not a camera measurement", runs["run1"]["fps_per_stream"], runs["run2"]["fps_per_stream"]],
                  ["≥30 FPS every source", "Sizing estimate only", runs["run1"]["status"], runs["run2"]["status"]],
                  ["DS efficiency", "100% baseline" if comparable else "N/A", efficiency(runs["run1"]), efficiency(runs["run2"])] ]),
              10: f'Run 1 efficiency: {efficiency(runs["run1"])}. Run 2 efficiency: {efficiency(runs["run2"])}. '
                  "Bottleneck attribution requires profiling; no unmeasured overhead breakdown is claimed."}
    result[8] += "\n\nGraph: " + data["scope"] + ". Rendering/encoding excluded. Every source is checked in the final five windows. "
    result[8] += "This is a tested stream count, not a proven absolute maximum. Rendered diagnostics are separate."
    if data.get('runner') == 'task-capacity':
        result[8] = table(['Run', 'Streams', 'Mux batch', 'Task batch / unit', 'Avg FPS/source',
                           'Worst sustained FPS/source', 'Aggregate frames/s', '≥30 FPS every source', '≥5 FPS every source'],
                          [[label, r['streams'], r['mux_batch'],
                            f"{r['stages'][-1]['inference_batch']} {r['stages'][-1]['batch_unit']}",
                            r['fps_per_stream'], r['worst_fps'], r['total_fps'], r['status'],
                            'PASS' if r['worst_fps'] >= 5 else 'FAIL']
                           for label, r in zip(['Calibration', 'Validation'], runs.values())])
        result[8] += '\n\nComplete upstream inference and task decoding are included; rendering/encoding are excluded. '
        result[8] += 'Every source is checked in five final windows. The tested count is not a proven absolute maximum. '
        result[8] += 'ROI batches are not camera counts. Caller acceptance at 5 FPS does not alter the 30-FPS result.'
        result[9] = result[9].replace('Batch / streams', 'TRT batch / measured DS streams')
    return result


def amend(markdown: Path, manifest: dict, root: Path) -> None:
    """Update generated status/validation reports, without touching legacy detection reports."""
    if "benchmark" not in manifest:
        return
    data = collect(manifest, root)  # invalid supplied evidence stops conversion
    replacements = sections(data)
    if data is not None and (data["input_unit"] == "frame" or data.get('runner') == 'task-capacity'):
        charts = markdown.parent / "charts"
        chart_data = markdown.parent / "verified-benchmark-data.json"
        chart_data.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
        subprocess.run([sys.executable, str(HERE / "generate-benchmark-charts.py"),
                        str(charts), str(chart_data)], check=True, timeout=90)
        for n, name in [(5, "trtexec_bs1_vs_bsmax"), (6, "trtexec_throughput"),
                        (8, "ds_streams_vs_fps"), (9, "trt_vs_ds"), (10, "efficiency")]:
            image = charts / f"chart_{name}.png"
            if not image.is_file() or image.stat().st_size == 0:
                raise ValueError("missing benchmark chart")
            replacements[n] += f'\n\n<div style="page-break-before: always;"></div>\n\n**{TITLES[n]}**\n\n![{TITLES[n]}](charts/{image.name})'
    text = markdown.read_text()
    if not all(re.search(rf"^## {n}\. ", text, re.MULTILINE) for n in TITLES):
        raise ValueError("benchmark report needs the twelve-section template")
    diagnostic = re.search(r"^### Separate task-capacity diagnostic\n.*?(?=^## \d+\. |\Z)", text, re.MULTILINE | re.DOTALL)
    if diagnostic:
        replacements[8] += "\n\n" + diagnostic[0].strip()
    # The legacy contract has five benchmark charts, not an extra timing chart.
    text = re.sub(r"!\[[^]]*\]\([^)]*chart_phase_timings\.png\)", "", text)
    for n, body in replacements.items():
        pattern = rf"^## {n}\. .*?(?=^## \d+\. |\Z)"
        text = re.sub(pattern, lambda _, n=n, body=body: f"## {n}. {TITLES[n]}\n\n{body}\n\n", text, flags=re.MULTILINE | re.DOTALL)
    markdown.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    manifest_path = Path(sys.argv[2]).resolve()
    amend(Path(sys.argv[1]), json.loads(manifest_path.read_text()), manifest_path.parent)
