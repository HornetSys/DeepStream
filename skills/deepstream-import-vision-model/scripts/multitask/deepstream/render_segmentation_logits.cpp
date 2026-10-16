// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// Type-level DeepStream renderer for one NCHW FP32 segmentation-logit head.
// Build against the installed DeepStream SDK. The caller supplies the model
// config, H.264 video, output dimensions, and class count; no model name is
// embedded. The class map is nearest-neighbor remapped to the mux frame before
// nvsegvisual, which otherwise paints a small native-resolution corner mask.

#include <gst/gst.h>
#include <glib.h>
#include <glib-unix.h>
#include <signal.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"

namespace {
GstElement* pipeline = nullptr;
GMainLoop* loop = nullptr;
std::ofstream records;
std::ofstream masks;
bool failed = false;
guint64 frames = 0;
guint64 mask_frames = 0;
guint64 mask_bytes = 0;
guint target_width = 0;
guint target_height = 0;
guint class_count = 0;
bool scalar_mode = false;
float scalar_min = 0.0f;
float scalar_max = 0.0f;

void release_mask(gpointer data, gpointer) {
  auto* user = static_cast<NvDsUserMeta*>(data);
  if (!user || !user->user_meta_data) return;
  auto* mask = static_cast<NvDsInferSegmentationMeta*>(user->user_meta_data);
  g_free(mask->class_map);
  g_free(mask->class_probabilities_map);
  g_free(mask);
}

gpointer copy_mask(gpointer data, gpointer) {
  auto* user = static_cast<NvDsUserMeta*>(data);
  if (!user || !user->user_meta_data) return nullptr;
  auto* source = static_cast<NvDsInferSegmentationMeta*>(user->user_meta_data);
  auto* copy = static_cast<NvDsInferSegmentationMeta*>(g_malloc0(sizeof(NvDsInferSegmentationMeta)));
  *copy = *source;
  const size_t count = static_cast<size_t>(source->width) * source->height;
  copy->class_map = static_cast<gint*>(g_malloc(count * sizeof(gint)));
  std::memcpy(copy->class_map, source->class_map, count * sizeof(gint));
  copy->class_probabilities_map = nullptr;
  copy->priv_data = nullptr;
  return copy;
}

void attach_mask(NvDsBatchMeta* batch, NvDsFrameMeta* frame,
                 const std::vector<gint>& native, guint width, guint height) {
  const size_t pixels = static_cast<size_t>(target_width) * target_height;
  auto* user = nvds_acquire_user_meta_from_pool(batch);
  auto* mask = static_cast<NvDsInferSegmentationMeta*>(g_malloc0(sizeof(NvDsInferSegmentationMeta)));
  mask->classes = class_count;
  mask->width = target_width;
  mask->height = target_height;
  mask->unique_id = 1;
  mask->class_map = static_cast<gint*>(g_malloc(pixels * sizeof(gint)));
  for (guint y = 0; y < target_height; ++y) {
    const guint sy = std::min(height - 1, static_cast<guint>(
        static_cast<guint64>(y) * height / target_height));
    for (guint x = 0; x < target_width; ++x) {
      const guint sx = std::min(width - 1, static_cast<guint>(
          static_cast<guint64>(x) * width / target_width));
      mask->class_map[static_cast<size_t>(y) * target_width + x] =
          native[static_cast<size_t>(sy) * width + sx];
    }
  }
  user->user_meta_data = mask;
  user->base_meta.meta_type = static_cast<NvDsMetaType>(NVDSINFER_SEGMENTATION_META);
  user->base_meta.copy_func = copy_mask;
  user->base_meta.release_func = release_mask;
  nvds_add_user_meta_to_frame(frame, user);
}

GstPadProbeReturn infer_probe(GstPad*, GstPadProbeInfo* info, gpointer) {
  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  NvDsBatchMeta* batch = buffer ? gst_buffer_get_nvds_batch_meta(buffer) : nullptr;
  if (!batch) return GST_PAD_PROBE_OK;
  for (NvDsMetaList* link = batch->frame_meta_list; link; link = link->next) {
    auto* frame = static_cast<NvDsFrameMeta*>(link->data);
    if (!frame) continue;
    ++frames;
    bool found = false;
    guint native_width = 0;
    guint native_height = 0;
    guint64 frame_mask_offset = mask_bytes;
    for (NvDsMetaList* meta_link = frame->frame_user_meta_list; meta_link;
         meta_link = meta_link->next) {
      auto* user = static_cast<NvDsUserMeta*>(meta_link->data);
      if (!user || user->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) continue;
      auto* tensor = static_cast<NvDsInferTensorMeta*>(user->user_meta_data);
      if (!tensor || tensor->num_output_layers != 1 || !tensor->out_buf_ptrs_host ||
          !tensor->out_buf_ptrs_host[0]) continue;
      auto* layer = &tensor->output_layers_info[0];
      if (layer->dataType != FLOAT) continue;
      NvDsInferDimsCHW dims = {0, 0, 0};
      getDimsCHWFromDims(dims, layer->inferDims);
      if (dims.c != (scalar_mode ? 1u : class_count) || dims.w == 0 || dims.h == 0 ||
          dims.w > 4096 || dims.h > 4096 ||
          layer->inferDims.numElements != dims.c * dims.w * dims.h) continue;
      const float* logits = static_cast<const float*>(tensor->out_buf_ptrs_host[0]);
      const size_t plane = static_cast<size_t>(dims.w) * dims.h;
      std::vector<gint> native(plane, -1);
      if (scalar_mode) {
        for (size_t pixel = 0; pixel < plane; ++pixel) {
          const float value = logits[pixel];
          if (!std::isfinite(value)) {
            failed = true;
            break;
          }
          const float fraction = std::clamp(
              (value - scalar_min) / (scalar_max - scalar_min), 0.0f, 1.0f);
          native[pixel] = std::min(static_cast<gint>(class_count - 1),
              static_cast<gint>(fraction * class_count));
        }
      } else {
        for (size_t pixel = 0; pixel < plane; ++pixel) {
          float best = -std::numeric_limits<float>::infinity();
          for (guint c = 0; c < class_count; ++c) {
            const float value = logits[static_cast<size_t>(c) * plane + pixel];
            if (std::isfinite(value) && value > best) {
              best = value;
              native[pixel] = static_cast<gint>(c);
            }
          }
        }
      }
      if (std::find(native.begin(), native.end(), -1) != native.end()) {
        failed = true;
        break;
      }
      masks.write(scalar_mode ? reinterpret_cast<const char*>(logits)
                              : reinterpret_cast<const char*>(native.data()),
                  static_cast<std::streamsize>(plane * sizeof(gint)));
      if (!masks) {
        failed = true;
        break;
      }
      mask_bytes += plane * sizeof(gint);
      attach_mask(batch, frame, native, dims.w, dims.h);
      native_width = dims.w;
      native_height = dims.h;
      found = true;
      ++mask_frames;
      break;
    }
    records << "{\"frame_num\":" << frame->frame_num
            << ",\"mask_present\":" << (found ? "true" : "false")
            << ",\"output_kind\":\"" << (scalar_mode ? "scalar_fp32" : "class_map_i32") << "\""
            << ",\"display_bins\":" << class_count;
    if (scalar_mode) {
      records << ",\"display_min\":" << scalar_min
              << ",\"display_max\":" << scalar_max;
    }
    records
            << ",\"native_width\":" << native_width
            << ",\"native_height\":" << native_height
            << ",\"mask_byte_offset\":" << frame_mask_offset
            << ",\"mask_byte_count\":" << (mask_bytes - frame_mask_offset)
            << ",\"render_width\":" << target_width
            << ",\"render_height\":" << target_height << "}\n";
  }
  return GST_PAD_PROBE_OK;
}

gboolean bus_callback(GstBus*, GstMessage* message, gpointer) {
  if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS) {
    if (scalar_mode) {
      g_print("SCALAR_EOS frames=%" G_GUINT64_FORMAT " maps=%" G_GUINT64_FORMAT "\n",
              frames, mask_frames);
    } else {
      g_print("SEGMENTATION_EOS frames=%" G_GUINT64_FORMAT " masks=%" G_GUINT64_FORMAT "\n",
              frames, mask_frames);
    }
    g_main_loop_quit(loop);
  } else if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
    GError* error = nullptr;
    gchar* debug = nullptr;
    gst_message_parse_error(message, &error, &debug);
    g_printerr("SEGMENTATION_ERROR: %s\n", error ? error->message : "unknown");
    g_clear_error(&error);
    g_free(debug);
    failed = true;
    g_main_loop_quit(loop);
  }
  return TRUE;
}

gboolean stop_request(gpointer) {
  if (pipeline) gst_element_send_event(pipeline, gst_event_new_eos());
  return G_SOURCE_CONTINUE;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 9 && argc != 12) {
    g_printerr("Usage: %s <config> <input.h264> <output.ogv> <structured.jsonl> <width> <height> <classes> <native-maps.raw> [--scalar <min> <max>]\n", argv[0]);
    return 2;
  }
  try {
    target_width = std::stoul(argv[5]);
    target_height = std::stoul(argv[6]);
    class_count = std::stoul(argv[7]);
    scalar_mode = argc == 12;
    if (scalar_mode) {
      if (std::strcmp(argv[9], "--scalar") != 0) return 2;
      scalar_min = std::stof(argv[10]);
      scalar_max = std::stof(argv[11]);
    }
  } catch (...) {
    g_printerr("Invalid dimensions or class count\n");
    return 2;
  }
  if (target_width < 16 || target_width > 4096 || target_height < 16 ||
      target_height > 4096 || class_count == 0 || class_count > 512 ||
      (scalar_mode && (class_count < 2 || !std::isfinite(scalar_min) ||
                       !std::isfinite(scalar_max) || scalar_min >= scalar_max)) ||
      !std::filesystem::is_regular_file(argv[1]) ||
      !std::filesystem::is_regular_file(argv[2]) ||
      std::filesystem::exists(argv[3]) || std::filesystem::exists(argv[4]) ||
      std::filesystem::exists(argv[8])) {
    g_printerr("Invalid config, input, frame geometry, class count, or existing output\n");
    return 2;
  }
  gst_init(&argc, &argv);
  records.open(argv[4], std::ios::out | std::ios::binary);
  masks.open(argv[8], std::ios::out | std::ios::binary);
  if (!records || !masks) return 2;
  gchar* input = g_strescape(argv[2], nullptr);
  gchar* config = g_strescape(argv[1], nullptr);
  gchar* output = g_strescape(argv[3], nullptr);
  gchar* graph = g_strdup_printf(
      "filesrc location=\"%s\" ! h264parse ! nvv4l2decoder ! "
      "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! mux.sink_0 "
      "nvstreammux name=mux batch-size=1 width=%u height=%u live-source=0 batched-push-timeout=40000 ! "
      "nvinfer name=infer config-file-path=\"%s\" ! "
      "nvsegvisual width=%u height=%u batch-size=1 original-background=true alpha=0.55 ! "
      "nvvideoconvert ! video/x-raw,format=I420 ! theoraenc quality=63 ! "
      "oggmux ! filesink location=\"%s\" sync=false",
      input, target_width, target_height, config,
      target_width, target_height, output);
  g_free(input);
  g_free(config);
  g_free(output);
  GError* parse_error = nullptr;
  pipeline = gst_parse_launch(graph, &parse_error);
  g_free(graph);
  if (!pipeline || parse_error) {
    g_printerr("Pipeline parse failed: %s\n", parse_error ? parse_error->message : "unknown");
    g_clear_error(&parse_error);
    if (pipeline) gst_object_unref(pipeline);
    return 3;
  }
  GstElement* infer = gst_bin_get_by_name(GST_BIN(pipeline), "infer");
  GstPad* pad = infer ? gst_element_get_static_pad(infer, "src") : nullptr;
  if (!pad) return 3;
  gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, infer_probe, nullptr, nullptr);
  gst_object_unref(pad);
  gst_object_unref(infer);
  loop = g_main_loop_new(nullptr, FALSE);
  const guint interrupt_watch = g_unix_signal_add(SIGINT, stop_request, nullptr);
  const guint terminate_watch = g_unix_signal_add(SIGTERM, stop_request, nullptr);
  GstBus* bus = gst_element_get_bus(pipeline);
  const guint watch = gst_bus_add_watch(bus, bus_callback, nullptr);
  gst_object_unref(bus);
  if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    failed = true;
  } else {
    g_main_loop_run(loop);
  }
  gst_element_set_state(pipeline, GST_STATE_NULL);
  records.close();
  masks.close();
  if (watch) g_source_remove(watch);
  if (interrupt_watch) g_source_remove(interrupt_watch);
  if (terminate_watch) g_source_remove(terminate_watch);
  gst_object_unref(pipeline);
  g_main_loop_unref(loop);
  return failed || frames == 0 || mask_frames != frames ? 1 : 0;
}
