/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DS3D_MULTISENSOR_MIXER_IMPL_H
#define DS3D_MULTISENSOR_MIXER_IMPL_H

#include "multisensor_mixer.h"

#include "ds3d/common/helper/memdata.h"
#include "ds3d/common/helper/signalshot.h"
#include "ds3d/common/hpp/datamixer.hpp"
#include "ds3d/common/impl/impl_datamixer.h"

#include "multisensor_mixer_config.h"
#include <chrono>
#include <ctime>

namespace ds3d { namespace impl { namespace mixer {

struct FrameDuration {
    uint32_t portId = 0;
    double frameDuration = 0.0;
};

class DataMapEntry {
public:
    using OnGuardDataCBImpl = std::function<void(ErrCode, GuardDataMap)>;
    DataMapEntry(GuardDataMap dm, OnGuardDataCBImpl iCb, int32_t portId)
    {
        datamap = dm;
        inputConsumedCb = iCb;
        _portId = portId;
    }
    int32_t getPortId() { return _portId; }
    OnGuardDataCBImpl inputConsumedCb;
    GuardDataMap datamap;
    TimeStamp ts;

private:
    int32_t _portId;
};

class MixerInputPort {
public:
    MixerInputPort(int32_t pId)
    {
        _portId = pId;
    }

    ErrCode enqueue(ShrdPtr<DataMapEntry> entry)
    {
        DS3D_ERROR_RETURN(
            entry->datamap.getData(kTimeStamp, entry->ts),
            "input datamap for port[%d] does not have timestamp; ignoring buffer", _portId);
        std::unique_lock<std::mutex> locker(_mtx);
        _inputQueue.emplace_back(std::move(entry));
        return ErrCode::kGood;
    }

    uint32_t capacity()
    {
        std::unique_lock<std::mutex> locker(_mtx);
        return _inputQueue.size();
    }

    // number of buffers in _inputQueue with timestamp <= desiredTimeStamp >= 1
    bool readyToMix(TimeStamp desiredTimeStamp)
    {
        std::unique_lock<std::mutex> locker(_mtx);
        if (_inputQueue.empty()) {
            return false;
        }
        // the buffers in queue are in the order of increasing timestamp
        ShrdPtr<DataMapEntry> e = _inputQueue.back();
        if (e->ts.t0 >= desiredTimeStamp.t0) {
            return true;
        }
        return false;
    }

    ShrdPtr<DataMapEntry> dequeue()
    {
        std::unique_lock<std::mutex> locker(_mtx);
        if (_inputQueue.empty()) {
            return nullptr;
        }
        ShrdPtr<DataMapEntry> e = _inputQueue.front();
        _inputQueue.pop_front();
        return e;
    }

    int32_t getPortId() { return _portId; }

    double calculateAvgFrameDuration(uint64_t ts)
    {
        _numOfInputBuffers++;
        if (_numOfInputBuffers == 1) {
            _firstBufferTimestamp = ts;
            return 0.0;
        }
        return (ts - _firstBufferTimestamp) / (_numOfInputBuffers - 1);
    }

    void flush()
    {
        std::unique_lock<std::mutex> locker(_mtx);
        while (!_inputQueue.empty()) {
            _inputQueue.pop_front();
        }
    }

private:
    int32_t _portId;
    std::deque<ShrdPtr<DataMapEntry>> _inputQueue;
    std::mutex _mtx;
    // Theoretical fps calculation based on buffer timestamps
    uint64_t _firstBufferTimestamp = 0;
    uint64_t _numOfInputBuffers = 0;
};


// using MemPtr = std::unique_ptr<MemData>;

class MultiSensorMixer : public BaseImplDataMixer {
public:
    MultiSensorMixer() = default;
    ~MultiSensorMixer() override;

protected:
    ErrCode processImpl(const int portId, GuardDataMap inputData, OnGuardDataCBImpl inputConsumedCb) override;
    ErrCode setOutputCbImpl(OnGuardDataCBImpl outputDataCb) override;
    ErrCode stopImpl() override;
    ErrCode flushImpl() override;
    ErrCode startImpl(const std::string& content, const std::string& path);

    ErrCode updateInputImpl(int portId, MixerUpdate updateType);

    enum class ItemType {
        kData,
        kEvent,
    };

    struct QueItem {
        ItemType type = ItemType::kData;
        std::function<ErrCode()> process;
        std::function<void()> failed;
    };

    ErrCode mixerThread();

    std::tuple<std::cv_status, uint64_t> waitForTimerOrInput(uint64_t timeout);
    std::tuple<uint32_t, uint32_t, uint32_t> checkReadiness(uint64_t timeElapsedInWaitSinceLastPush);
    ErrCode mix();

private:
    Config _config;
    std::unordered_map<int, ShrdPtr<MixerInputPort>> _ports;
    ShrdPtr<std::thread> _thread;
    OnGuardDataCBImpl _outputDataCb;
    std::mutex _mtx;
    // condition_var for timed wait
    std::condition_variable _cv;
    // Time at which the last mixed output was pushed out of mixer
    std::chrono::time_point<std::chrono::steady_clock> _lastPushTime;
    std::chrono::time_point<std::chrono::steady_clock> _lastAlertTime;
    TimeStamp _lastPushTimestamp = {0};
    bool _mixedAtleastOneBuffer = false;
    std::atomic<bool> _stopMixer;
    std::atomic<ErrCode> _mixerThreadStatus;
    bool _flushing = false;

private:
    uint32_t portsPopulated();
    std::tuple<uint32_t, uint32_t, uint32_t> getPortsReadyToMix(TimeStamp desiredTimeStamp);
    // force flush buffers in a port without mixing
    void flushPort(int32_t portId);
    void checkPerformance();
    std::multimap<uint64_t, ShrdPtr<DataMapEntry>> _mixerInputMap;
    FrameDuration _slowestStream;
    FrameDuration _fastestStream;
    SignalShot _shot;
};

}}}  // namespace ds3d::impl::mixer


#endif  // DS3D_MULTISENSOR_MIXER_IMPL_H
