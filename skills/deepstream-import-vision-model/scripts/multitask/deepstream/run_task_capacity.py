#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Measure a caller-owned task graph on one GPU without rendering or encoding.

This benchmark is not an adapter promotion or a semantic accuracy validator.
Native parsers or a caller-approved tensor decoder must execute before counting
each source's frames. ROI inference batches are recorded separately from camera
count. No capacity is extrapolated from source zero or TensorRT throughput.
"""
from __future__ import annotations

import argparse
import configparser
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import run_type_adapter_pipeline as tensors

TASKS = {'detection', 'classification', 'segmentation', 'pose', 'depth'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_plan(plan):
    if plan.get('task') not in TASKS:
        raise ValueError('capacity runner supports the five vision task categories')
    for key in ('streams', 'mux_batch', 'width', 'height'):
        if type(plan.get(key)) is not int or plan[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if plan['streams'] > 512 or plan['mux_batch'] > plan['streams']:
        raise ValueError('streams must be <=512 and mux_batch <= streams')
    if not 16 <= plan['width'] <= 4096 or not 16 <= plan['height'] <= 4096:
        raise ValueError('mux dimensions must be in [16,4096]')
    if not 60 <= plan.get('duration_seconds', 60) <= 300:
        raise ValueError('duration must be 60 through 300 seconds')
    if not 0 <= plan.get('warmup_seconds', 10) <= 30:
        raise ValueError('warmup must be at most 30 seconds')
    if not math.isfinite(float(plan.get('target_fps', 30))) or float(plan.get('target_fps', 30)) <= 0:
        raise ValueError('target FPS must be finite and positive')
    media = Path(plan['video']).resolve(strict=True)
    if not media.is_file() or not media.stat().st_size:
        raise ValueError('video must be a nonempty local regular file')
    stages = plan.get('stages')
    if not isinstance(stages, list) or not stages:
        raise ValueError('stages must declare the complete ordered inference graph')
    ids = set()
    records = []
    for index, stage in enumerate(stages):
        config = Path(stage['config']).resolve(strict=True)
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(config)
        prop = parser['property']
        stage_id = int(prop['gie-unique-id'])
        if stage_id <= 0 or stage_id in ids or int(prop.get('interval', 0)) != 0:
            raise ValueError('inference IDs must be unique and interval must be zero')
        mode = int(prop['process-mode'])
        if (index == 0 and mode != 1) or (index > 0 and mode != 2):
            raise ValueError('first stage must be primary; remaining stages must be secondary')
        if index > 0 and int(prop.get('operate-on-gie-id', 0)) not in ids:
            raise ValueError('secondary stage must refer to an earlier detector')
        if int(prop.get('classifier-async-mode', 0)) or int(prop.get('secondary-reinfer-interval', 0)):
            raise ValueError('benchmark requires synchronous per-frame secondary inference')
        ids.add(stage_id)
        batch = int(prop['batch-size'])
        if batch < 1 or batch > 512:
            raise ValueError('inference batch must be in [1,512]')
        engine = Path(prop['model-engine-file'])
        if not engine.is_absolute():
            engine = config.parent / engine
        engine = engine.resolve(strict=True)
        if not engine.is_file():
            raise ValueError('engine must be an existing caller-owned artifact')
        records.append({'config': str(config), 'config_sha256': digest(config),
                        'engine': str(engine), 'engine_sha256': digest(engine),
                        'component_id': stage_id, 'inference_batch': batch,
                        'batch_unit': 'frames' if mode == 1 else 'ROIs'})
    if plan.get('decoder_kind') not in {'native-detection', 'native-classification', 'tensor-hook'}:
        raise ValueError('an executable decoder_kind is required')
    if plan['decoder_kind'] == 'tensor-hook' and not plan.get('decoder'):
        raise ValueError('tensor routes require a caller-owned decoder file.py:decode')
    expected = {'native-detection': '0', 'native-classification': '1', 'tensor-hook': '100'}
    if prop.get('network-type') != expected[plan['decoder_kind']]:
        raise ValueError('final inference network type does not match decoder_kind')
    return media, records


def summarize(windows, streams, target):
    recent = windows[-5:]
    per_source = []
    for source_id in range(streams):
        values = [window['fps'][source_id] for window in recent]
        per_source.append({'source_id': source_id, 'windows_fps': values,
                           'minimum_fps': min(values) if values else None,
                           'average_fps': sum(values)/len(values) if values else None})
    passed = len(recent) == 5 and all(v['minimum_fps'] >= target for v in per_source)
    return {'status': 'PASS' if passed else 'FAIL', 'window_count': len(windows),
            'per_stream': per_source, 'streams': streams, 'target_fps_per_stream': target,
            'aggregate_average_fps': sum(v['average_fps'] or 0 for v in per_source)}


def load_decoder(target):
    filename, symbol = target.rsplit(':', 1)
    if symbol != 'decode':
        raise ValueError('tensor decoder must name the decode hook')
    path = Path(filename).resolve(strict=True)
    spec = importlib.util.spec_from_file_location('caller_capacity_decoder', path)
    if spec is None or spec.loader is None:
        raise ValueError('decoder must be a loadable Python module')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    decode = getattr(module, 'decode', None)
    if not callable(decode):
        raise TypeError('decoder is not callable')
    return decode, {'path': str(path), 'sha256': digest(path)}


def run(plan, output):
    media, stages = validate_plan(plan)
    runner_sha256 = digest(Path(__file__).resolve())
    if output.exists():
        raise ValueError('capacity output directory already exists; use a fresh attempt')
    output.mkdir(parents=True)
    import gi
    gi.require_version('Gst', '1.0')
    import pyds
    from gi.repository import GLib, Gst
    Gst.init(None)
    hook, hook_record = load_decoder(plan['decoder']) if plan['decoder_kind'] == 'tensor-hook' else (None, None)
    pipeline = Gst.Pipeline.new('task-capacity')
    mux = Gst.ElementFactory.make('nvstreammux', 'mux')
    sink = Gst.ElementFactory.make('fakesink', 'sink')
    if mux is None or sink is None:
        raise ValueError('missing nvstreammux or fakesink')
    mux.set_property('batch-size', plan['mux_batch'])
    mux.set_property('width', plan['width'])
    mux.set_property('height', plan['height'])
    mux.set_property('batched-push-timeout', 40000)
    sink.set_property('sync', False)
    sink.set_property('async', False)
    pipeline.add(mux)
    pipeline.add(sink)
    errors = []
    def source_pad(_source, pad, source_id):
        caps = pad.get_current_caps() or pad.query_caps(None)
        print('SOURCE_PAD', source_id, caps.to_string() if caps else 'missing-caps', flush=True)
        if not caps or not caps.get_structure(0).get_name().startswith('video/'):
            return
        target = mux.request_pad_simple(f'sink_{source_id}')
        if target is None or pad.link(target) != Gst.PadLinkReturn.OK:
            errors.append(f'failed source link: {source_id}')
        else:
            print('SOURCE_LINKED', source_id, flush=True)
    for source_id in range(plan['streams']):
        source = Gst.ElementFactory.make('nvurisrcbin', f'source_{source_id}')
        if source is None:
            raise ValueError('missing nvurisrcbin')
        source.set_property('uri', media.as_uri())
        source.set_property('file-loop', True)
        source.set_property('disable-audio', True)
        source.set_property('gpu-id', 0)
        pipeline.add(source)
        source.connect('pad-added', source_pad, source_id)
    last = mux
    for index, stage in enumerate(stages):
        queue = Gst.ElementFactory.make('queue', f'queue_{index}')
        infer = Gst.ElementFactory.make('nvinfer', f'infer_{index}')
        if queue is None or infer is None:
            raise ValueError('missing inference or queue element')
        infer.set_property('config-file-path', stage['config'])
        pipeline.add(queue)
        pipeline.add(infer)
        if not last.link(queue) or not queue.link(infer):
            raise ValueError('failed inference stage link')
        last = infer
    if not last.link(sink):
        raise ValueError('failed fakesink link')
    counts = [0] * plan['streams']
    tasks = [0] * plan['streams']
    samples = {}
    component = stages[-1]['component_id']
    def probe(_pad, info):
        try:
            buffer = info.get_buffer()
            batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer)) if buffer else None
            if batch is None:
                raise ValueError('capacity buffer has no frame metadata')
            for frame in tensors._iter_meta(batch.frame_meta_list, pyds.NvDsFrameMeta.cast):
                source_id = int(frame.source_id)
                if source_id < 0 or source_id >= plan['streams']:
                    raise ValueError('unexpected source ID')
                decoded = []
                if hook:
                    owners = ([frame] if len(stages) == 1 else list(tensors._iter_meta(frame.obj_meta_list, pyds.NvDsObjectMeta.cast)))
                    for owner in owners:
                        user_list = owner.frame_user_meta_list if len(stages) == 1 else owner.obj_user_meta_list
                        for user in tensors._iter_meta(user_list, pyds.NvDsUserMeta.cast):
                            if user.base_meta.meta_type != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                                continue
                            tensor = pyds.NvDsInferTensorMeta.cast(user.user_meta_data)
                            if int(tensor.unique_id) != component:
                                continue
                            if int(tensor.num_output_layers) <= 0:
                                raise ValueError('tensor metadata has no output layers')
                            values = dict(tensors._layer_array(pyds, tensor, i) for i in range(int(tensor.num_output_layers)))
                            record = hook(values, {'source_id': source_id, 'frame_id': int(frame.frame_num)})
                            if not isinstance(record, dict) or not record:
                                raise ValueError('decoder produced no structured task record')
                            json.dumps(record, allow_nan=False)
                            decoded.append(record)
                    if len(stages) == 1 and not decoded:
                        raise ValueError('full-frame tensor metadata is missing')
                else:
                    for obj in tensors._iter_meta(frame.obj_meta_list, pyds.NvDsObjectMeta.cast):
                        if plan['decoder_kind'] == 'native-detection' and int(obj.unique_component_id) == component:
                            box = [float(getattr(obj.rect_params, k)) for k in ('left', 'top', 'width', 'height')]
                            if not all(math.isfinite(v) for v in box) or box[2] <= 0 or box[3] <= 0:
                                raise ValueError('detector emitted invalid coordinates')
                            score = float(obj.confidence)
                            if not math.isfinite(score):
                                raise ValueError('detector emitted non-finite confidence')
                            decoded.append({'class_id': int(obj.class_id), 'bbox': box, 'score': score})
                        elif plan['decoder_kind'] == 'native-classification':
                            for classifier in tensors._iter_meta(obj.classifier_meta_list, pyds.NvDsClassifierMeta.cast):
                                if int(classifier.unique_component_id) != component:
                                    continue
                                for label in tensors._iter_meta(classifier.label_info_list, pyds.NvDsLabelInfo.cast):
                                    text = str(label.result_label)
                                    if not text or len(text) > 256 or any(ord(c) < 32 or ord(c) > 126 for c in text):
                                        raise ValueError('classifier emitted invalid label')
                                    score = float(label.result_prob)
                                    if not math.isfinite(score):
                                        raise ValueError('classifier emitted non-finite score')
                                    decoded.append({'label': text, 'score': score})
                counts[source_id] += 1
                tasks[source_id] += len(decoded)
                if decoded and source_id not in samples:
                    samples[source_id] = decoded[0]
        except Exception as exc:
            if len(errors) < 10:
                errors.append(str(exc))
        return Gst.PadProbeReturn.OK
    sink.get_static_pad('sink').add_probe(Gst.PadProbeType.BUFFER, probe)
    bus = pipeline.get_bus()
    windows = []
    start = time.monotonic()
    warmup = plan.get('warmup_seconds', 10)
    duration = plan.get('duration_seconds', 60)
    boundary = start + warmup
    previous = counts.copy()
    previous_tasks = tasks.copy()
    startup = pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE
    try:
        while startup and time.monotonic() < start + warmup + duration and not errors:
            # nvurisrcbin's file-loop seeks use GLib callbacks. A bus-only
            # polling loop leaves short fixtures stalled after their first EOS.
            context = GLib.MainContext.default()
            callbacks = 0
            while context.pending() and callbacks < 100 and time.monotonic() < start + warmup + duration:
                context.iteration(False)
                callbacks += 1
            message = bus.timed_pop_filtered(100 * Gst.MSECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if message:
                if message.type == Gst.MessageType.ERROR:
                    error, debug = message.parse_error()
                    errors.append(f'{error}: {debug}')
                else:
                    errors.append('unexpected EOS before the bounded capacity interval completed')
                break
            now = time.monotonic()
            if now < start + warmup:
                previous = counts.copy()
                previous_tasks = tasks.copy()
                continue
            if now - boundary >= 2:
                current = counts.copy()
                fps = [(current[i]-previous[i])/(now-boundary) for i in range(plan['streams'])]
                current_tasks = tasks.copy()
                windows.append({'duration_seconds': now-boundary, 'fps': fps,
                                'task_records': [current_tasks[i]-previous_tasks[i] for i in range(plan['streams'])]})
                print('TASK_CAPACITY', json.dumps(windows[-1]), flush=True)
                boundary, previous = now, current
                previous_tasks = current_tasks
    finally:
        pipeline.set_state(Gst.State.NULL)
    result = summarize(windows, plan['streams'], float(plan.get('target_fps', 30)))
    result.update(schema_version=1, producer='deepstream-task-capacity-runner', task=plan['task'],
                  campaign_id=plan.get('campaign_id', ''), startup=startup, errors=errors,
                  frames_per_source=counts, task_records_per_source=tasks, decoder=hook_record,
                  mux_batch=plan['mux_batch'], stages=stages, video_sha256=digest(media),
                  runner_sha256=runner_sha256,
                  windows=windows, samples=samples, duration_seconds=time.monotonic()-start,
                  scope='decode-mux-complete-inference-cascade-task-postprocessing-fakesink',
                  rendering_included=False, encoding_included=False,
                  semantic_accuracy_claim=False, ceiling_claim='tested stream count, not an exhaustive maximum')
    if not startup or errors or not all(tasks) or any(not all(w['task_records']) for w in windows[-5:]):
        result['status'] = 'FAIL'
    with (output / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    with (output / 'plan.json').open('x') as stream:
        json.dump(plan, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in {'windows', 'samples'}}, indent=2))
    return 0 if result['status'] == 'PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path, help='caller-approved graph and decoder contract JSON')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(json.loads(args.plan.read_text()), args.output_dir)
    except (ImportError, OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({'status': 'BLOCKED', 'reason': str(exc)}), file=sys.stderr)
        return 4


if __name__ == '__main__':
    raise SystemExit(main())
