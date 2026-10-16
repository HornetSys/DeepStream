/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nvdsgstutils.h"
#include "gst-nvquery-internal.h"
#include "string.h"

#include <gst/rtp/gstrtcpbuffer.h>

#define NTP_DATA_KEY "nvdsgstutils-ntp-data"

#define SLIDING_WIN_PTS_LEN 16
#define SLIDING_WIN_PTS_PERCENT 70

static GMutex ntp_data_lock;

//WINDOWS
#ifdef _WIN32
#include "windows.h"
void nvdsgstutils_init (void);
void nvdsgstutils_deinit (void);

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved)
{
  switch (ul_reason_for_call)
  {
      case DLL_PROCESS_ATTACH:
          nvdsgstutils_init();
          break;
      case DLL_PROCESS_DETACH:
          nvdsgstutils_deinit();
          break;
  }
  return TRUE;
}
#else
void __attribute__ ((constructor)) nvdsgstutils_init (void);
void __attribute__ ((destructor)) nvdsgstutils_deinit (void);
#endif
typedef struct
{
  GstClockTime prev_frame_pts;
  GstClockTime pts_window[SLIDING_WIN_PTS_LEN];
  GstClockTime prev_packet_pts;
  gboolean is_first_pkt;
  guint current_index;
  GMutex data_lock;
} _PtsData;

//WINDOWS
#ifdef _WIN32
void nvdsgstutils_init (void)
#else
void __attribute__ ((constructor)) nvdsgstutils_init (void)
#endif
{
    g_mutex_init (&ntp_data_lock);
}
//WINDOWS
#ifdef _WIN32
void nvdsgstutils_deinit (void)
#else
void __attribute__ ((destructor)) nvdsgstutils_deinit (void)
#endif
{
    g_mutex_clear (&ntp_data_lock);
}

static gboolean
parse_ntp_values_from_rtp_sync_struct (GstStructure * s,
        GstClockTime *ntp_time_epoch_ns, GstClockTime *frame_timestamp)
{
  gboolean more;
  GstRTCPPacket packet;
  guint64 ntptime = 0;
  guint32 rtptime;
  GstBuffer *buffer;
  guint64 gstreamer_time;
  guint64 base_rtptime;
  guint64 sr_ext_rtptime;
  guint clock_rate;
  GstRTCPBuffer rtcp = { NULL, };

  if (!s)
    return FALSE;

  if (G_UNLIKELY(!gst_structure_has_name (s, "application/x-rtp-sync")))
    return FALSE;

  if (!gst_structure_has_field (s, "sr-buffer"))
    return FALSE;

  buffer =
      gst_value_get_buffer (gst_structure_get_value (s, "sr-buffer"));
  gstreamer_time =
      g_value_get_uint64 (gst_structure_get_value (s, "base-time"));
  base_rtptime =
      g_value_get_uint64 (gst_structure_get_value (s, "base-rtptime"));
  sr_ext_rtptime =
      g_value_get_uint64 (gst_structure_get_value (s, "sr-ext-rtptime"));
  clock_rate =
      g_value_get_uint (gst_structure_get_value (s, "clock-rate"));

  if (clock_rate == 0)
    return FALSE;

  gstreamer_time += gst_util_uint64_scale (sr_ext_rtptime - base_rtptime, GST_SECOND, clock_rate);

  if (!buffer) {
    return FALSE;
  }

  gst_rtcp_buffer_map (buffer, GST_MAP_READ, &rtcp);

  for (more = gst_rtcp_buffer_get_first_packet (&rtcp, &packet); more;
      more = gst_rtcp_packet_move_to_next (&packet)) {
    switch (gst_rtcp_packet_get_type (&packet)) {
      case GST_RTCP_TYPE_SR:

        /* get NTP and RTP times */
        gst_rtcp_packet_sr_get_sender_info (&packet, NULL, &ntptime, &rtptime,
            NULL, NULL);

        /** rtptime - which is the timestamp synchronized with corresponding
         * RTP Stream is dropped as we have the same mapped to Gstreamer clock
         * in rtpbin (manager) plugin which is eventually saved in
         * gstreamer_time as "base-time".
         * NOTE: This is the latest incoming RTP-buffer-time aligned to
         * gstreamer clock + clock-skew between receiver and sender
         * FOR OSS REFERENCE: Specific code in rtpmanager:
         * do_handle_sync(), rtp_jitter_buffer_get_sync() in
         * gst-plugins-good/gst/rtpmanager/gstrtpjitterbuffer.c
         */
        (void) rtptime;

        /** RTCP RFC 3550; The full-resolution
         * NTP timestamp is a 64-bit unsigned fixed-point number with
         * the integer part in the first 32 bits and the fractional part in the
         * last 32 bits.
         * The NTP timescale wraps around every 2^32 seconds (136 years);
         * the first rollover will occur in 2036.
         * The higher 32-bits carry epoch time + 2208988800LL (later is a const introduced by gstrtpbin.c)
         * To convert fractional part of NTP:
         * NTP fraction * (fraction of seconds) / 2 ^ 32.
         * For example of NTP fraction 1329481807, to convert to microsecond:
         * = 1329481807 * (10 ^ 6) / 2 ^ 32 = 309544us (roughly)
         */
        {
          struct timespec timespec_rtcp_ntp;
            /** Extract higher 32-bit epoch into timespec_rtcp_ntp.tv_sec */
          GST_TIME_TO_TIMESPEC ((((ntptime >> 32) - 2208988800LL) * GST_SECOND),
              timespec_rtcp_ntp);
            /** Extract lower 32-bit fraction into timespec_rtcp_ntp.tv_nsec */
          timespec_rtcp_ntp.tv_nsec =
              (guint64) (((ntptime & (0xFFFFFFFF)) * GST_SECOND) >> 32);

          *ntp_time_epoch_ns = GST_TIMESPEC_TO_TIME (timespec_rtcp_ntp);
          *frame_timestamp = gstreamer_time;
          gst_rtcp_buffer_unmap (&rtcp);
          return TRUE;
        }

        break;
      default:
        break;
    }
  }
  gst_rtcp_buffer_unmap (&rtcp);
  return FALSE;
}

static GstClockTime
get_average_frame_time (_PtsData * pts_data)
{
  guint pts_cnt = 0, i = 0, j = 0;
  GstClockTime avg_frame_pts[SLIDING_WIN_PTS_LEN - 1];
  GstClockTime frame_pts_window[SLIDING_WIN_PTS_LEN];
  GstClockTime frame_pts = 0, a = 0;

  g_mutex_lock (&pts_data->data_lock);
  memcpy(frame_pts_window, pts_data->pts_window, (sizeof(GstClockTime) * SLIDING_WIN_PTS_LEN));
  g_mutex_unlock (&pts_data->data_lock);

  /* sort the pts array */
  for (i = 0; i < SLIDING_WIN_PTS_LEN; i++) {
    for (j = i + 1; j < SLIDING_WIN_PTS_LEN; j++) {
      if (frame_pts_window[i] > frame_pts_window[j]) {
        a =  frame_pts_window[i];
        frame_pts_window[i] = frame_pts_window[j];
        frame_pts_window[j] = a;
      }
    }
  }
  /* create diff array from pts array */
  for(i = 0; i < SLIDING_WIN_PTS_LEN - 1; i++) {
    avg_frame_pts[i] = frame_pts_window[i + 1] - frame_pts_window[i];
  }

  /* Check if sliding window is full. No zero values should be present. */
  for (i = 0; i < SLIDING_WIN_PTS_LEN - 1; i++) {
    if (avg_frame_pts[i] == 0)
      return 0;
  }

  /* sort the array and average over 0.8 *SLIDING_WIN_PTS_LEN length */
  for (i = 0; i < SLIDING_WIN_PTS_LEN - 1; i++) {
    for (j = i + 1; j < SLIDING_WIN_PTS_LEN - 1; j++) {
      if (avg_frame_pts[i] > avg_frame_pts[j]) {
        a =  avg_frame_pts[i];
        avg_frame_pts[i] = avg_frame_pts[j];
        avg_frame_pts[j] = a;
      }
    }
  }

  for (i = 0; i < SLIDING_WIN_PTS_LEN - 1; i++) {
    if(avg_frame_pts[i] != 0) {
      frame_pts+= avg_frame_pts[i];
      pts_cnt++;
      if (pts_cnt >= ((SLIDING_WIN_PTS_LEN - 1) * SLIDING_WIN_PTS_PERCENT) / 100)
        break;
    }
  }
  if(pts_cnt > 0) {
    frame_pts = frame_pts / pts_cnt;
  } else {
    frame_pts = 0;
  }
  return frame_pts;
}

static void
deepstream_rtp_bin_handle_sync (GstElement * jitterbuffer, GstStructure * s,
    gpointer * user_data)
{
  GstClockTime ntp_time_epoch_ns, frame_timestamp;
  if (!parse_ntp_values_from_rtp_sync_struct (s, &ntp_time_epoch_ns,
          &frame_timestamp)) {
    return;
  }

  _NtpData *data = (_NtpData *) g_malloc (sizeof (_NtpData));
  data->frame_timestamp = frame_timestamp;
  data->ntp_time_epoch_ns = ntp_time_epoch_ns;
  data->avg_frame_time = get_average_frame_time ((_PtsData *) user_data);

  GstQuery *query = gst_nvquery_ntp_sync_new ();
  gst_nvquery_ntp_sync_set (query, data);

  GstPad *src_pad = gst_element_get_static_pad (jitterbuffer, "src");
  gst_pad_peer_query (src_pad, query);
  gst_object_unref (src_pad);
  gst_query_unref (query);

  g_mutex_lock (&ntp_data_lock);
  g_object_set_data_full (G_OBJECT (jitterbuffer), NTP_DATA_KEY, data,
          (GDestroyNotify) g_free);
  g_mutex_unlock (&ntp_data_lock);
}

static GstPadProbeReturn
handle_ntp_sync_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstObject *jitterbuffer = GST_OBJECT_PARENT (pad);

  if (info->type & GST_PAD_PROBE_TYPE_QUERY_UPSTREAM) {
    GstQuery *query = (GstQuery *) info->data;
    g_mutex_lock (&ntp_data_lock);
    _NtpData *data = (_NtpData *) g_object_get_data (G_OBJECT (jitterbuffer), NTP_DATA_KEY);
    _NtpData ldata = { 0, 0, 0 };
    if (!data)
      data = &ldata;
    g_mutex_unlock (&ntp_data_lock);
    data->avg_frame_time = get_average_frame_time ((_PtsData *) u_data);

    if (gst_nvquery_is_ntp_sync (query) && data) {
      gst_nvquery_ntp_sync_set (query, data);
    }
  }
  return GST_PAD_PROBE_OK;
}

  static GstPadProbeReturn
handle_jitterbuffer_output_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  _PtsData *pts_data = (_PtsData *)u_data;

  if (info->type & GST_PAD_PROBE_TYPE_BUFFER) {
    GstClockTime buf_pts = GST_BUFFER_PTS(info->data);

    if (G_UNLIKELY (pts_data->is_first_pkt)) {
      pts_data->prev_packet_pts = buf_pts;
      pts_data->prev_frame_pts = buf_pts;
      pts_data->is_first_pkt = FALSE;
    }

    if(pts_data->prev_packet_pts != buf_pts) {
      g_mutex_lock (&pts_data->data_lock);
      pts_data->pts_window[pts_data->current_index] = buf_pts;
      pts_data->prev_frame_pts = buf_pts;

      pts_data->current_index++;
      g_mutex_unlock (&pts_data->data_lock);
      pts_data->prev_packet_pts = buf_pts;
    }

    if(pts_data->current_index == SLIDING_WIN_PTS_LEN) {
      pts_data->current_index = 0;
    }
  }

  return GST_PAD_PROBE_OK;
}

static void
release_pts_data (gpointer data)
{
  _PtsData *pts_data = (_PtsData *) data;
  g_mutex_clear (&pts_data->data_lock);
  g_free (pts_data);
}

static _PtsData *
pts_data_new (void)
{
  _PtsData *pts_data = (_PtsData *) g_malloc0 (sizeof(_PtsData));
  pts_data->is_first_pkt = TRUE;
  g_mutex_init (&pts_data->data_lock);
  return pts_data;
}

static void
deep_element_added_callback (GstBin * self, GstBin * sub_bin,
    GstElement * element, gpointer user_data)
{
  GstElementFactory *factory = gst_element_get_factory (element);
  const gchar *factory_name = gst_plugin_feature_get_name (factory);


  if (!g_strcmp0 (factory_name, "rtpjitterbuffer")) {
    _PtsData *pts_data = pts_data_new ();
    g_object_set (G_OBJECT (element), "mode", 4, "max-rtcp-rtp-time-diff", -1, "faststart-min-packets", 1,NULL);
    g_signal_connect (G_OBJECT (element), "handle-sync",
        G_CALLBACK (deepstream_rtp_bin_handle_sync), pts_data);
    GstPad *pad = gst_element_get_static_pad (element, "src");
    gst_pad_add_probe (pad, GST_PAD_PROBE_TYPE_QUERY_UPSTREAM,
        handle_ntp_sync_probe, pts_data, NULL);
    gst_pad_add_probe (pad, GST_PAD_PROBE_TYPE_BUFFER,
        handle_jitterbuffer_output_probe, pts_data, NULL);
    g_object_set_data_full (G_OBJECT(element), "nv-pts-data", pts_data, release_pts_data);
    g_object_unref(pad);
  }
}

void
configure_source_for_ntp_sync (GstElement * src_elem)
{
  GstElementFactory *factory = gst_element_get_factory (src_elem);
  const gchar *factory_name = gst_plugin_feature_get_name (factory);

  if (!g_strcmp0 (factory_name, "rtspsrc")
      || !g_strcmp0 (factory_name, "uridecodebin")) {
    g_signal_connect (G_OBJECT (src_elem), "deep-element-added",
        G_CALLBACK (deep_element_added_callback), GST_ELEMENT_NAME (src_elem));
  } else {
    g_printerr ("WARNING: %s: Source element type %s is not supported",
        __func__, factory_name);
  }
  return;
}
