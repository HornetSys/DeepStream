################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

#!/usr/bin/env python3
"""
Simple script to extract embeddings from DeepStream pipeline using probe
"""

import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import json
import numpy as np

sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
import pyds

class EmbeddingExtractor:
    def __init__(self, image_path, engine_path, output_path):
        self.image_path = image_path
        self.engine_path = engine_path
        self.output_path = output_path
        self.embeddings = []
        self.loop = None
        
    def probe_handler(self, pad, info):
        """Probe to extract embeddings from metadata"""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
            
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
                
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                    
                # Look for ReID metadata
                l_user = obj_meta.obj_user_meta_list
                while l_user is not None:
                    try:
                        user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                        if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_TRACKER_OBJ_REID_META:
                            reid_meta = pyds.NvDsObjReid.cast(user_meta.user_meta_data)
                            if reid_meta and reid_meta.ptr_host:
                                # Extract embedding
                                emb = []
                                for i in range(reid_meta.featureSize):
                                    emb.append(reid_meta.ptr_host[i])
                                
                                print(f"\n[EMBEDDING FOUND] Dimension: {reid_meta.featureSize}")
                                emb_array = np.array(emb)
                                print(f"  L2 norm: {np.linalg.norm(emb_array):.6f}")
                                print(f"  Range: [{emb_array.min():.6f}, {emb_array.max():.6f}]")
                                print(f"  First 10: {emb_array[:10].tolist()}")
                                
                                self.embeddings.append({
                                    'object_id': obj_meta.object_id,
                                    'class_id': obj_meta.class_id,
                                    'embedding': emb
                                })
                    except StopIteration:
                        break
                    try:
                        l_user = l_user.next
                    except StopIteration:
                        break
                        
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
                    
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
                
        return Gst.PadProbeReturn.OK
    
    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("\n[PIPELINE] End of stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"\n[ERROR] {err}: {debug}")
            loop.quit()
        elif t == Gst.MessageType.STATE_CHANGED:
            if isinstance(message.src, Gst.Pipeline):
                old, new, pending = message.parse_state_changed()
                print(f"[PIPELINE] State: {old.value_nick} -> {new.value_nick}")
        return True
    
    def run(self):
        Gst.init(None)
        
        print("=" * 70)
        print("DeepStream Embedding Extractor (Python)")
        print("=" * 70)
        print(f"Image: {self.image_path}")
        print(f"Engine: {self.engine_path}")
        print(f"Output: {self.output_path}\n")
        
        # Create pipeline
        pipeline_str = (
            f"filesrc location={self.image_path} ! "
            f"jpegdec ! videoconvert ! video/x-raw,format=RGBA ! "
            f"nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA ! "
            f"nvstreammux name=mux batch-size=1 width=1920 height=1080 ! "
            f"nvinfer config-file-path=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt ! "
            f"nvdsvisionencoder backend=tensorrt tensorrt-engine={self.engine_path} batch-size=16 verbose=true ! "
            f"fakesink"
        )
        
        print("[PIPELINE] Creating...")
        print(f"{pipeline_str}\n")
        
        pipeline = Gst.parse_launch(pipeline_str)
        
        # Add probe to fakesink
        sink = pipeline.get_by_name("fakesink0")
        if sink:
            sinkpad = sink.get_static_pad("sink")
            if sinkpad:
                sinkpad.add_probe(Gst.PadProbeType.BUFFER, self.probe_handler, 0)
        
        # Setup bus
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        self.loop = GLib.MainLoop()
        bus.connect("message", self.bus_call, self.loop)
        
        # Start pipeline
        print("[PIPELINE] Starting...\n")
        pipeline.set_state(Gst.State.PLAYING)
        
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        
        # Cleanup
        print("\n[CLEANUP] Stopping...")
        pipeline.set_state(Gst.State.NULL)
        
        # Save embeddings
        if self.embeddings:
            print(f"\n[SUCCESS] Found {len(self.embeddings)} embeddings")
            
            output_data = {
                'backend': 'tensorrt_deepstream_python',
                'engine': self.engine_path,
                'image_path': self.image_path,
                'embeddings': []
            }
            
            for emb_info in self.embeddings:
                emb_array = np.array(emb_info['embedding'])
                output_data['embeddings'].append({
                    'object_id': emb_info['object_id'],
                    'class_id': emb_info['class_id'],
                    'dimension': len(emb_array),
                    'l2_norm': float(np.linalg.norm(emb_array)),
                    'min_value': float(emb_array.min()),
                    'max_value': float(emb_array.max()),
                    'embedding_vector': emb_array.tolist()
                })
            
            with open(self.output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"[SAVED] {self.output_path}")
        else:
            print("\n[WARNING] No embeddings captured")
            return 1
        
        print("=" * 70)
        return 0

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <image> <engine.plan> [output.json]", file=sys.stderr)
        sys.exit(2)
    image = sys.argv[1]
    engine = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else 'deepstream_python_embeddings.json'

    extractor = EmbeddingExtractor(image, engine, output)
    sys.exit(extractor.run())
