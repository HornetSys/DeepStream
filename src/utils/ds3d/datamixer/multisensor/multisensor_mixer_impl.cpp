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

#include "multisensor_mixer_impl.h"
#include "ds3d/common/impl/impl_frames.h"
#include <gst/gst.h>
#include "nvbufsurface.h"
#include "nvdsmeta.h"
#include "gstnvdsmeta.h"

#include <unistd.h>
#include <thread>

namespace ds3d { namespace impl { namespace mixer {

using NanoSecondsType = std::chrono::duration<double, std::nano>;
using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<std::chrono::steady_clock, NanoSecondsType>;

static inline bool
readyToMix(
    std::cv_status waitStatus, uint32_t portsReadyToMix, uint32_t portsWithBuffers, uint32_t portsSize, bool forceSync)
{
    if (portsReadyToMix == 0) {
        return false;
    }

    if (portsReadyToMix == portsSize) {
        return true;
    }

    if (forceSync) {
        return false;
    }

    return ((waitStatus == std::cv_status::timeout) && (portsWithBuffers > 0));
}

void
MultiSensorMixer::checkPerformance()
{
    auto now = Clock::now();
    if (_slowestStream.frameDuration && _fastestStream.frameDuration &&
        (now - _lastPushTime) > NanoSecondsType(_slowestStream.frameDuration * 2) &&
        (now - _lastAlertTime) > std::chrono::duration<double>(10)) {
        LOG_WARNING(
            "mixer is running slower than expected (slowest/fastest stream(:%d/%d) "
            "duration(=%lfms/%lfms)); timeout setting (%lfms) may be incorrect",
            _slowestStream.portId, _fastestStream.portId, _slowestStream.frameDuration * kNasToMs,
            _fastestStream.frameDuration * kNasToMs, _config.frameTimeout * kNasToMs);
        _lastAlertTime = Clock::now();
    }
}

ErrCode
MultiSensorMixer::mixerThread()
{
    uint64_t timeout = (uint64_t)(_config.frameTimeout);
    uint64_t timeElapsedInWaitSinceLastPush = 0;
    do {
        std::cv_status waitStatus;
        uint64_t timeElapsedInWait;
        std::tie(waitStatus, timeElapsedInWait) = waitForTimerOrInput(timeout);
        timeElapsedInWaitSinceLastPush += timeElapsedInWait;
        checkPerformance();
        while (true) {
            uint32_t portsReadyToMix = 0;
            uint32_t portsWithBuffers = 0;
            uint32_t portsSize = 0;
            std::tie(portsReadyToMix, portsWithBuffers, portsSize) = checkReadiness(timeElapsedInWaitSinceLastPush);
            if (readyToMix(waitStatus, portsReadyToMix, portsWithBuffers, portsSize, _config.forceSync) ||
                (portsWithBuffers && _flushing)) {
                LOG_DEBUG(
                    "ready to mix; portsReadyToMix(%d), portsSize(%d), portsWithBuffers(%d) isFlushing=%d",
                    portsReadyToMix, portsSize, portsWithBuffers, _flushing);
                _mixerThreadStatus = mix();
                DS3D_ERROR_RETURN(
                    _mixerThreadStatus.load(), "mixer thread exiting; error =%d", (int)_mixerThreadStatus.load());
                if (timeout > timeElapsedInWait) {
                    timeout = timeout - timeElapsedInWait;
                } else {
                    timeout = 0;
                }
                if (_mixerInputMap.empty()) {
                    _shot.signal();
                }
                waitStatus = std::cv_status::no_timeout;
                timeElapsedInWaitSinceLastPush = 0;
                continue;  // continue to check and mix all ready buffers
            } else {
                timeout = uint64_t(_config.frameTimeout);
            }
            break;  // break and wait
        };
    } while (!_stopMixer.load());
    return ErrCode::kGood;
}

/**
 * wait for timer to expire or an input to arrive for one of the ports
 * return == std::cv_status::timeout if timeout expired
 *        == std::cv_status::no_timeout if an input arrived before timeout
 */
std::tuple<std::cv_status, uint64_t>
MultiSensorMixer::waitForTimerOrInput(uint64_t timeout)
{
    std::cv_status waitStatus = std::cv_status::no_timeout;
    TimePoint now = Clock::now();
    waitStatus = std::cv_status::no_timeout;
    NanoSecondsType waitTime = NanoSecondsType(0);
    if (_lastPushTime + NanoSecondsType(timeout) > now) {
        waitTime = _lastPushTime + NanoSecondsType(timeout) - now;
    } else if (!portsPopulated()) {
        waitTime = NanoSecondsType(timeout);
    }
    if (waitTime > NanoSecondsType(0)) {
        std::unique_lock<std::mutex> locker(_mtx);
        waitStatus = _cv.wait_for(locker, waitTime);
    }

    return std::make_tuple(waitStatus, (uint64_t)(Clock::now() - now).count());
}

/**
 * return tuple with
 * 1) number of ports with atleast one buffer with (timestamp <= desiredTimeStamp)
 * 2) number of ports with atleast one buffer
 * 3) portsSize; the number of current active ports
 */
std::tuple<uint32_t, uint32_t, uint32_t>
MultiSensorMixer::getPortsReadyToMix(TimeStamp desiredTimeStamp)
{
    uint32_t portsWithReadyBuffers = 0;
    uint32_t portsWithBuffers = 0;
    std::unique_lock<std::mutex> locker(_mtx);
    for (auto it : _ports) {
        ShrdPtr<MixerInputPort> port = it.second;
        if (port && port->capacity()) {
            portsWithBuffers++;
            if (port->readyToMix(desiredTimeStamp)) {
                portsWithReadyBuffers++;
            }
        }
    }
    LOG_DEBUG(
        "ready to mix?; portsReadyToMix(%d), portsSize(%lu), portsWithBuffers(%d)", portsWithReadyBuffers,
        _ports.size(), portsWithBuffers);
    return std::make_tuple(portsWithReadyBuffers, portsWithBuffers, _ports.size());
}

// number of ports with atleast one buffer available
uint32_t
MultiSensorMixer::portsPopulated()
{
    std::unique_lock<std::mutex> locker(_mtx);
    uint32_t portsWithReadyBuffers = 0;
    for (auto it : _ports) {
        ShrdPtr<MixerInputPort> port = it.second;
        if (port && port->capacity()) {
            portsWithReadyBuffers++;
        }
    }
    return portsWithReadyBuffers;
}

std::tuple<uint32_t, uint32_t, uint32_t>
MultiSensorMixer::checkReadiness(uint64_t timeElapsedInWaitSinceLastPush)
{
    uint64_t hop = (uint64_t)(_config.frameTimeout);
    if (hop > timeElapsedInWaitSinceLastPush) {
        hop = hop - timeElapsedInWaitSinceLastPush;
    } else {
        hop = 0;
    }
    TimeStamp desiredNext = _lastPushTimestamp;
    desiredNext.t0 += hop;
    return getPortsReadyToMix(desiredNext);
}

/**
 * Pick one buffer from each port and mix
 */
ErrCode
MultiSensorMixer::mix()
{
    GuardDataMap datamap;
    bool mixed = false;
    TimeStamp timestamp = {0};
    std::vector<ShrdPtr<DataMapEntry>> consumedList;
    ErrCode ret = ErrCode::kGood;

    std::unique_lock<std::mutex> locker(_mtx);
    std::set<int32_t> portsToMix;
    auto it = _mixerInputMap.begin();
    LOG_DEBUG("mixing: {");
    while (it != _mixerInputMap.end()) {
        auto portId = it->second->getPortId();
        if (_ports.find(portId) == _ports.end()) {
            // port unavailable
            break;
        }
        auto port = _ports[portId];
        if (portsToMix.insert(portId).second == false) {
            // this portId was already mixed; portId just repeated
            break;
        }
        // we can mix this as this portId was never mixed
        if (!mixed) {
            datamap.reset(NvDs3d_CreateDataHashMap());
            mixed = true;
        }
        // pick one buffer
        auto portDataMapEntry = port->dequeue();
        it = _mixerInputMap.erase(it);
        if (portDataMapEntry) {
            LOG_DEBUG("mixing [%d:%lu]", portId, portDataMapEntry->ts.t0);
            if (portDataMapEntry->ts.t0 > timestamp.t0) {
                timestamp = portDataMapEntry->ts;
            }
            char portIdString[1024];
            snprintf(portIdString, 1024, "%d", portId);
            ErrCode ok = datamap.copy(portDataMapEntry->datamap, DataMapPolicy::kCopyWithIntegerId, portIdString);
            if (!isGood(ok)) {
                ret = ok;
                LOG_ERROR("datamap copy failed (err=%d) in data mixer", (int)ok);
            }
            // release locker is not safe during iteration, inputConsumedCb call later to prevent
            // hung issue
            consumedList.emplace_back(portDataMapEntry);
        }
    }
    LOG_DEBUG("mixing done }");
    _lastPushTime = Clock::now();
    _lastPushTimestamp = timestamp;
    _mixedAtleastOneBuffer = true;

    locker.unlock();
    // callback all entries together after locker released
    for (auto& entry : consumedList) {
        assert(entry);
        if (entry->inputConsumedCb) {
            entry->inputConsumedCb(ErrCode::kGood, entry->datamap);
        }
    }
    consumedList.clear();

    if (!mixed) {
        return ErrCode::kGood;
    }

    datamap.setData(kTimeStamp, timestamp);
    LOG_DEBUG("mixer output:");
    datamap.printDebug();
    /** push output */
    _outputDataCb(ret, datamap);

    return ErrCode::kGood;
}

MultiSensorMixer::~MultiSensorMixer()
{
    stop_i();
}

ErrCode
MultiSensorMixer::processImpl(const int portId, GuardDataMap inputData, OnGuardDataCBImpl inputConsumedCb)
{
    LOG_DEBUG("datamixer is starting to mixer datamap");

    if (_mixerThreadStatus != ErrCode::kGood) {
        LOG_ERROR("mixer thread exited; error = %d; ignoring input data at mixer", (int)_mixerThreadStatus.load());
        return _mixerThreadStatus;
    }
    LOG_DEBUG("mixer input for port (%d)", portId);
    inputData.printDebug();

    auto entry = ShrdPtr<DataMapEntry>(new DataMapEntry(inputData, inputConsumedCb, portId));

    std::unique_lock<std::mutex> locker(_mtx);
    if (_ports.find(portId) == _ports.end()) {
        locker.unlock();
        LOG_DEBUG("port undefinied; automatically adding port with updateInput() API for mixer portId(%d)", portId);
        ErrCode ok = updateInputImpl(portId, MixerUpdate::kPortAdded);
        DS3D_ERROR_RETURN(
            ok, "port undefinied; automatically adding port with updateInput() API for mixer portId(%d) failed",
            portId);
        locker.lock();
    }

    if (_flushing) {
        locker.unlock();
        LOG_WARNING("ignoring input data at mixer; mixer is flushing");
        inputConsumedCb(ErrCode::kGood, inputData);
        return ErrCode::kGood;
    }

    if (!_mixedAtleastOneBuffer) {
        if (entry->ts.t0 > _lastPushTimestamp.t0) {
            _lastPushTimestamp = entry->ts;
        }
    }
    ErrCode ok = _ports[portId]->enqueue(entry);
    double avgFrameDuration = _ports[portId]->calculateAvgFrameDuration(entry->ts.t0);
    _mixerInputMap.insert(std::pair<uint64_t, ShrdPtr<DataMapEntry>>(entry->ts.t0, entry));
    if (_slowestStream.frameDuration < avgFrameDuration) {
        _slowestStream.frameDuration = avgFrameDuration;
        _slowestStream.portId = portId;
    }
    if (!_fastestStream.frameDuration || _fastestStream.frameDuration > avgFrameDuration) {
        _fastestStream.frameDuration = avgFrameDuration;
        _fastestStream.portId = portId;
    }
    locker.unlock();
    if (!isGood(ok)) {
        inputConsumedCb(ok, inputData);
    }
    DS3D_ERROR_RETURN(ok, "enqueue for portId(%d) failed", portId);

    _cv.notify_all();

    return ErrCode::kGood;
}

ErrCode
MultiSensorMixer::setOutputCbImpl(OnGuardDataCBImpl oCb)
{
    _outputDataCb = oCb;
    return ErrCode::kGood;
}

ErrCode
MultiSensorMixer::stopImpl()
{
    LOG_INFO("datamixer is closing");

    _flushing = false;
    _stopMixer = true;
    _cv.notify_all();
    if (_thread) {
        _thread->join();
        _thread.reset();
    }
    return ErrCode::kGood;
}

ErrCode
MultiSensorMixer::flushImpl()
{
    std::unique_lock<std::mutex> locker(_mtx);
    uint32_t times = 0;
    LOG_INFO("datamixer is flushing");
    _flushing = true;
    while (!_mixerInputMap.empty() && times < 10) {
        locker.unlock();
        LOG_DEBUG("datamixer is flushing; buffers in queue=%lu", _mixerInputMap.size());
        _shot.wait((_config.frameTimeout * kNasToMs) * 10);
        times++;
        locker.lock();
    }
    if (times == 10) {
        LOG_WARNING("datamixer flush failed\n");
    }
    _flushing = false;
    LOG_INFO("datamixer flushing completed");
    // TODO; make sure flushing completed
    return ErrCode::kGood;
}

ErrCode
MultiSensorMixer::startImpl(const std::string& content, const std::string& path)
{
    LOG_INFO("datamixer is starting");
    ErrCode code = config::CatchYamlCall([&, this]() { return parseConfig(content, path, _config); });
    DS3D_ERROR_RETURN(code, "parse datamixer config: %s failed", path.c_str());
    setInputCaps(_config.compConfig.gstInCaps);
    setOutputCaps(_config.compConfig.gstOutCaps);

    _mixerThreadStatus = ErrCode::kGood;
    _stopMixer = false;
    _flushing = false;
    _lastPushTime = Clock::now();
    _lastAlertTime = Clock::now();
    _thread = ShrdPtr<std::thread>(new std::thread(&MultiSensorMixer::mixerThread, this));

    return ErrCode::kGood;
}

ErrCode
MultiSensorMixer::updateInputImpl(int portId, MixerUpdate updateType)
{
    std::unique_lock<std::mutex> locker(_mtx);
    // Create MixerInputPort and add to _ports[]
    auto port = _ports.find(portId);
    if ((updateType == MixerUpdate::kPortAdded) && (port == _ports.end())) {
        _ports[portId] = ShrdPtr<MixerInputPort>(new MixerInputPort(portId));
    } else if ((updateType == MixerUpdate::kPortRemoved) && (port != _ports.end())) {
        // TODO; clean all buffers for this port from port->queue and multimap
        locker.unlock();
        flushPort(portId);
        locker.lock();
        _ports.erase(port);
    }
    return ErrCode::kGood;
}

void
MultiSensorMixer::flushPort(int32_t portId)
{
    std::unique_lock<std::mutex> locker(_mtx);
    auto port = _ports.find(portId);
    if (port == _ports.end()) {
        return;
    }
    _ports[portId]->flush();
    std::vector<ShrdPtr<DataMapEntry>> flusingList;
    auto it = _mixerInputMap.begin();
    while (it != _mixerInputMap.end()) {
        if (it->second->getPortId() == portId) {
            flusingList.emplace_back(it->second);
            it = _mixerInputMap.erase(it);
        } else {
            it++;
        }
    }
    locker.unlock();

    // callback all entries together after locker released
    for (auto& entry : flusingList) {
        assert(entry);
        if (entry->inputConsumedCb) {
            entry->inputConsumedCb(ErrCode::kGood, entry->datamap);
        }
    }
}

}}}  // namespace ds3d::impl::mixer

using namespace ds3d;

DS3D_EXTERN_C_BEGIN
DS3D_EXPORT_API abiRefDataMixer*
createMultiSensorMixer()
{
    return NewAbiRef<abiDataMixer>(new impl::mixer::MultiSensorMixer);
}

DS3D_EXTERN_C_END
