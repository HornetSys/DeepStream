/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nvds3d_gst_plugin.h"

#include "ds3d/common/common.h"
#include "ds3d/common/hpp/dataloader.hpp"
#include "ds3d/common/hpp/datamap.hpp"
#include "gst-nvevent.h"
#include "gst-nvmessage.h"
#include "gstnvdsmeta.h"
#include "nvds3d_meta.h"

using namespace ds3d;

using SamplePtr = gst::GstMiniObjPtr<GstSample>;

static void
StopAndDestroyDataLoader(gpointer dataloader)
{
    GuardDataLoader loader((abiRefDataLoader*)dataloader, true);  // take owner
    if (loader) {
        loader.stop();
    }
}

static void
NotifySrcError(GstAppSrc* src, ErrCode code, const char* msg)
{
    if (code < ErrCode::kGood) {
        LOG_ERROR("dataloder error: %s", msg);
        GST_ELEMENT_ERROR(
            src, STREAM, FAILED,  // GST_STREAM_ERROR_FAILED
            ("appsrc Failed to stream due to ds3d errors"), ("appsrc error: %s", msg));
    }
}

static void CheckAppSrcFlowError(GstAppSrc* src, GstFlowReturn flowRet) {
    if (flowRet >= GST_FLOW_OK)
        return;
    if (flowRet == GST_FLOW_FLUSHING || flowRet == GST_FLOW_EOS) {
        return;
    } else {
        NotifySrcError(src, ErrCode::kGst, "appsrc failed to push buffer to downstream plugins");
    }
}

static void
AppsrcFillData(GstAppSrc* src, guint length, gpointer userData)
{
    GuardDataLoader loader((abiRefDataLoader*)(userData), false);
    DS_ASSERT(loader);
    GuardDataMap datamap;
    ErrCode c = ErrCode::kGood;

    if (loader.state() == State::kNone) {
        DataProcessUserData* uData = (DataProcessUserData*)loader.getUserData();
        if (!uData || uData->pluginMagicUD != kDataProcessUserDataMagic) {
            NotifySrcError(
                src, ErrCode::kUnknown,
                "dataloader's userdata is not found or magic ID doesn't match, it might not set by "
                "appsrc");
            return;
        }
        c = loader.start(uData->configContent, uData->configPath);
        if (!isGood(c)) {
            NotifySrcError(src, c, "dataloader starts failed inside AppsrcFillData");
            return;
        }
    }
    if (loader.state() != State::kRunning) {
        NotifySrcError(
            src, ErrCode::kState, "Appsrc fill data failed since dataloader is not running.");
        return;
    }
    c = loader.readData(datamap);
    if (c < ErrCode::kGood) {
        NotifySrcError(src, c, "Read data from dataloader failed");
        return;
    }
    if (c == ErrCode::kByPass) {
        return;
    }
    bool endOfStream = (c == ErrCode::KEndOfStream);

    if (datamap) {
        GstBuffer* buf = nullptr;
        c = NvDs3D_CreateGstBuf(buf, datamap.abiRef(), false);
        if (!isGood(c)) {
            NotifySrcError(src, c, "Create ds3d datamap buffer failed");
            return;
        }
        DS_ASSERT(buf);

        bool needSendNvEvent = false;
        c = datamap.getData(kFirstSourceFrame, needSendNvEvent);

        if (isGood(c) && needSendNvEvent) {
            LOG_INFO("source send GST_NVEVENT_PAD_ADDED to pad!");
            uint32_t sourceId = 0;
            datamap.getData(kSourceId, sourceId);
            GstPad *SrcPad = gst_element_get_static_pad ((GstElement *)src, "src");
            GstEvent *new_pad_event = gst_nvevent_new_pad_added(sourceId);
            gst_pad_push_event(SrcPad, new_pad_event);
        }

        TimeStamp ts{0};
        c = datamap.getData(kTimeStamp, ts);
        if(isGood(c)) {
            LOG_DEBUG("set gstbuffer pts %lld", (long long int)ts.t0);
            GST_BUFFER_PTS(buf) = ts.t0;
        }

        GstFlowReturn flowRet = gst_app_src_push_buffer(src, buf);
        CheckAppSrcFlowError(src, flowRet);
        if (flowRet < GST_FLOW_OK) {
            endOfStream = true;
        }
    }

    if (endOfStream) {
        GstFlowReturn flowRet = gst_app_src_end_of_stream(src);
        CheckAppSrcFlowError(src, flowRet);
    }

    if (!datamap) {
        LOG_DEBUG("No data read from loader, might be EOS reached");
    }
}

static void
AppsrcEnoughData(GstAppSrc* src, gpointer user_data)
{
    return;
}

static gboolean
AppsrcSeekData(GstAppSrc* src, guint64 offset, gpointer user_data)
{
    return FALSE;
}


static void
StopAndDestroyDataRender(gpointer dataloader)
{
    GuardDataRender render((abiRefDataRender*)dataloader, true);  // take owner
    if (render) {
        render.stop();
    }
}

static void
NotifySinkError(GstAppSink* sink, ErrCode code, const char* msg)
{
    if (code < ErrCode::kGood) {
        LOG_ERROR("datarender error: %s", msg);
        GST_ELEMENT_ERROR(
            sink, STREAM, FAILED,  // GST_STREAM_ERROR_FAILED
            ("appsink Failed to stream due to ds3d errors"), ("appsink error: %s", msg));
    }
}

static void
AppsinkEos(GstAppSink* sink, gpointer user_data)
{
    GuardDataRender render((abiRefDataRender*)user_data, false);
    DS_ASSERT(render);
    if (render.state() == State::kRunning) {
        ErrCode c = render.flush();
        if (!isGood(c)) {
            NotifySinkError(sink, c, "");
        }
    }
}

#define SinkCheckError(boolExpr, fallback, fmt, ...) \
    do {                                             \
        if(boolExpr) break;                          \
        char __msg[1024] = {0};                      \
        snprintf(__msg, 1023, fmt, ##__VA_ARGS__);   \
        NotifySinkError(sink, ErrCode::kGst, __msg);          \
        fallback;                                    \
    } while (0)

#define SinkReturnFlowError(errStatement, fmt, ...) \
    SinkCheckError((errStatement) >= ErrCode::kGood, return GST_FLOW_ERROR, fmt, ##__VA_ARGS__)

static GstFlowReturn
AppsinkPreroll(GstAppSink* sink, gpointer user_data)
{
    DS_ASSERT(sink);
    GuardDataRender render((abiRefDataRender*)user_data, false);
    DS_ASSERT(render);
    if (render.state() != State::kRunning) {
        NotifySinkError(
            sink, ErrCode::kState, "Appsink preroll failed since datarender is not running.");
        return GST_FLOW_ERROR;
    }

    ErrCode c = ErrCode::kGood;
    SamplePtr sample;
    sample.reset(gst_app_sink_pull_preroll(sink));
    if (!sample) {
        if (gst_app_sink_is_eos(sink)) {
            c = render.flush();
            SinkReturnFlowError(c, "appsink EOS flush failed");
            return GST_FLOW_EOS;
        }
        return GST_FLOW_OK;
    }
    GstBuffer* buf = gst_sample_get_buffer(sample);
    const ds3d::abiRefDataMap* datamap = nullptr;
    c = NvDs3D_Find1stDataMap(buf, datamap);
    SinkReturnFlowError(c, "no datamap found in gstbuffer during appsink rendering");

    DS_ASSERT(datamap);
    GuardDataMap guardData(*datamap);
    c = render.preroll(guardData);
    SinkReturnFlowError(c, "datarender preroll data failed.");

    return GST_FLOW_OK;
}

static GstFlowReturn
AppsinkRender(GstAppSink* sink, gpointer user_data)
{
    GuardDataRender render((abiRefDataRender*)user_data, false);
    DS_ASSERT(render);
    SinkCheckError(
        render.state() == State::kRunning, return GST_FLOW_ERROR,
        "Appsink render failed since datarender is not running.");

    ErrCode c = ErrCode::kGood;
    SamplePtr sample;
    sample.reset(gst_app_sink_pull_sample(sink));
    if (!sample) {
        if (gst_app_sink_is_eos(sink)) {
            c = render.flush();
            SinkReturnFlowError(c, "appsink EOS flush failed");
            return GST_FLOW_EOS;
        }
        return GST_FLOW_OK;
    }
    GstBuffer* buf = gst_sample_get_buffer(sample);
    const ds3d::abiRefDataMap* datamap = nullptr;
    c = NvDs3D_Find1stDataMap(buf, datamap);
    SinkReturnFlowError(c, "no datamap found in gstbuffer during appsink rendering");

    DS_ASSERT(datamap);
    GuardDataMap guardData(*datamap);
    c = render.render(guardData, [sink](ErrCode err, const abiRefDataMap*) {
        SinkCheckError(err >= ErrCode::kGood, , "render callback report failure");
    });
    SinkReturnFlowError(c, "datarender rendering data failed.");
    return GST_FLOW_OK;
}

DS3D_EXTERN_C_BEGIN

DS3D_EXPORT_API ErrCode
NvDs3D_GstAppSrcSetDataloader(GstAppSrc* appSrc, abiRefDataLoader* refLoader)
{
    GstAppSrcCallbacks callbacks = {
        AppsrcFillData,
        AppsrcEnoughData,
        AppsrcSeekData,
    };

    DS_ASSERT(refLoader);
    GuardDataLoader loader(refLoader, true);
    loader.setErrorCallback([appSrc](ErrCode c, const char* str) { NotifySrcError(appSrc, c, str); });
    gst_app_src_set_callbacks(appSrc, &callbacks, loader.release(), StopAndDestroyDataLoader);

    return ErrCode::kGood;
}

ErrCode NvDs3D_GstAppSinkSetDataRender(
    GstAppSink* sink, ds3d::abiRefDataRender* refRender)
{
    GstAppSinkCallbacks callbacks = {
        AppsinkEos,
        AppsinkPreroll,
        AppsinkRender,
    };

    DS_ASSERT(refRender);
    GuardDataRender render(refRender, true);
    render.setErrorCallback([sink](ErrCode c, const char* str) { NotifySinkError(sink, c, str); });
    gst_app_sink_set_callbacks(sink, &callbacks, render.release(), StopAndDestroyDataRender);
    return ErrCode::kGood;
}

DS3D_EXTERN_C_END
