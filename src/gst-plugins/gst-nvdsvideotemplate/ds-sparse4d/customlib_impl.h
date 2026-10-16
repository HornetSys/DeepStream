/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CUSTOMLIB_IMPL_H
#define CUSTOMLIB_IMPL_H

// PyTorch Library
#include <torch/torch.h>

// NVIDIA DeepStream Library
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gst-nvquery.h"
#include "gstnvdsmeta.h"
#include "gst-nvevent.h"
#include "nvdscustomusermeta.h"
#include "nvdsdummyusermeta.h"
#include "nvdscustomlib_base.hpp"

// Preprocess Library
#include "preprocess/calibration_parser.h"

// Sparse4D Utils Library
#include "utils/tensor.h"
#include "utils/config_parser.h"
#include "utils/log_utils.h"

// Sparse4D Infer Library
#include "infer/engine_builder.h"
#include "infer/tensor_rt_engine.h"

// Format definitions
#define FORMAT_NV12 "NV12"
#define FORMAT_RGBA "RGBA"
#define FORMAT_I420 "I420"
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"

/* This quark is required to identify NvDsMeta when iterating through the buffer metadatas */
extern GQuark _dsmeta_quark;

// Type definitions for Eigen matrices
using Matrix4f = Eigen::Matrix<float, 4, 4>;

// Function declarations
void *set_metadata_ptr(void);
void fill_dummy_batch_meta_on_buffer(NvDsBatchMeta *batch_meta);

/**
 * @brief Sample implementation of the custom library
 */
class SampleAlgorithm : public DSCustomLibraryBase
{
public:
    /**
     * @brief Constructor
     */
    SampleAlgorithm()  {
        m_VectorProperty.clear();
        m_CustomConfigLoaded = false;
        m_SensorNames.clear();

        // Initialize logger pointer
        m_Logger = new PrettyLogger();

        // Initialize tensor cache
        m_TensorCache = new TensorCache();

        Initialize();
    }

    /**
     * @brief Initialize the custom library
     */
    void Initialize();

    /**
     * @brief Set initialization parameters
     *
     * @param params Creation parameters
     * @return true if successful, false otherwise
     */
    virtual bool SetInitParams(DSCustom_CreateParams *params);

    /**
     * @brief Set custom properties of the library
     *
     * @param prop Property to set
     * @return true if successful, false otherwise
     */
    virtual bool SetProperty(Property &prop);

    /**
     * @brief Handle GStreamer events
     *
     * @param event GStreamer event
     * @return true if successful, false otherwise
     */
    virtual bool HandleEvent(GstEvent *event);

    /**
     * @brief Query properties
     * 
     * @return Char pointer to properties string
     */
    virtual char *QueryProperties();

    /**
     * @brief Process incoming buffer
     *
     * @param inbuf Input GStreamer buffer
     * @return BufferResult Result of processing
     */
    virtual BufferResult ProcessBuffer(GstBuffer *inbuf);

    /**
     * @brief Get compatible caps
     *
     * @param direction Pad direction
     * @param in_caps Input caps
     * @param othercaps Other caps
     * @return GstCaps* Compatible caps
     */
    virtual GstCaps *GetCompatibleCaps(GstPadDirection direction,
        GstCaps* in_caps, GstCaps* othercaps);

    /**
     * @brief Destructor
     */
    ~SampleAlgorithm();

    gboolean hw_caps; /**< Hardware capabilities flag */

private:
    std::string errorMessage; /**< Error message storage for GOTO_ERROR macro */

    std::string skipMessage; /**< Skip message storage for GOTO_SKIP macro */

    /**
     * @brief Extract batch metadata from buffer
     *
     * @param buffer GStreamer buffer
     * @return NvDsBatchMeta* Batch metadata
     */
    NvDsBatchMeta *GetNVDS_BatchMeta(GstBuffer *buffer);


public:
    guint source_id = 0;                   /**< Source ID */
    guint m_FrameNum = 0;                  /**< Frame number (increments for all frames including skipped) */
    guint m_ProcessedFrameNum = 0;         /**< Processed frame number (only increments for actually processed frames) */
    guint m_DumpedFrameNum = 0;            /**< Dumped frame number */
    bool m_TransformMode = false;          /**< Transform mode flag */

    /* Custom Library Bufferpool */
    GstBufferPool *m_dsBufferPool = NULL;  /**< DeepStream buffer pool */
    GstBufferPool *m_swbufpool = NULL;     /**< Software buffer pool */
    guint swbuffersize;                    /**< Software buffer size */

    NvBufSurfTransformConfigParams m_ConfigParams;  /**< Transform config parameters */

    void *m_scratchNvBufSurface = NULL;     /**< Scratch NvBufSurface */

    // Currently dumps first 5 input video frame into file for demonstration purpose
    // Use vooya or similar player to view NV12 / RGBA video raw frame
    int dump_max_frames = 5;               /**< Maximum frames to dump */

    /* Additional flags for metadata insertion */
    bool m_dummyMetaInsert = false;        /**< Dummy metadata insertion flag */
    bool m_fillDummyBatchMeta = false;     /**< Fill dummy batch metadata flag */

    /* Config file parser properties */
    ConfigProperties m_ConfigProperties;   /**< Structure to store all configuration properties loaded from YAML config file */
    std::unique_ptr<ConfigParser> m_configParser;  /**< Config parser */
    bool m_CustomConfigLoaded = false;     /**< Custom config loaded flag */
    std::vector<Property> m_VectorProperty; /**< Vector containing key:value pair of custom lib properties */

    /* Logger instance */
    PrettyLogger* m_Logger;

    /* Vector containing sensor names */
    std::vector<std::string> m_SensorNames;

    /* Tensor cache */
    TensorCache* m_TensorCache;

    // TensorRT Engine instance for inference
    std::unique_ptr<TensorRTEngine> m_TensorRTEngine;

    /* Map to store projection matrices by camera name */
    std::map<std::string, Matrix4f> m_CameraProjectionMatrices;

    /* Map to store transformed projection matrices by camera name */
    std::map<std::string, Matrix4f> m_CameraTransformedProjectionMatrices;

    /* Flag to track if EOS has been received */
    bool m_eosReceived = false;
};

/**
 * @brief Create a custom algorithm context
 *
 * @param params Creation parameters
 * @return IDSCustomLibrary* Custom library instance
 */
extern "C" IDSCustomLibrary *CreateCustomAlgoCtx(DSCustom_CreateParams *params);

#endif /* CUSTOMLIB_IMPL_H */