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

#ifndef _DS3D_COMMON_IMPL_DATAHASHMAP__H
#define _DS3D_COMMON_IMPL_DATAHASHMAP__H

#include <ds3d/common/hpp/obj.hpp>
#include <unordered_map>

#include "ds3d/common/hpp/datamap.hpp"

namespace ds3d { namespace impl {

class DataHashTable : public abiDataMap {
public:
    DataHashTable() = default;
    ~DataHashTable() override
    {
        LOG_DEBUG("DataHashTable is destroying");
        removeAll();
    }

protected:
    struct DataInfo {
        TIdType tid = 0;
        abiRefAny* ud = nullptr;
    };
    using KeyName = std::string;

    bool has_i(const char* key) const
    {
        LockMutex locker(_mutex);
        return _hashTable.count(cppString(key));
    }

    ErrCode setBuf_i(const char* key, TIdType tid, abiRefAny* data) override
    {
        LockMutex locker(_mutex);
        return setBufUnsafe(key, tid, data);
    }

    ErrCode setBufUnsafe(const char* key, TIdType tid, abiRefAny* data)
    {
        DataInfo info{tid, data};
        DS_ASSERT(key);
        std::string name(key);

        auto iter = _hashTable.find(name);
        ErrCode code = ErrCode::kGood;
        if (iter != _hashTable.end()) {
                DataInfo& update = iter->second;
                removeDataInfo(update);
                iter->second = info;
        } else {
            code = _hashTable.emplace(name, info).second ? ErrCode::kGood : ErrCode::kUnknown;
        }
        return code;
    }
    ErrCode getBuf_i(const char* key, TIdType tid, const abiRefAny*& data) const override
    {
        DS_ASSERT(key);
        std::string name(key);
        LockMutex locker(_mutex);
        auto iter = _hashTable.find(name);
        if (iter == _hashTable.end()) {
            return ErrCode::kNotFound;
        }
        const DataInfo& found = iter->second;
        if (found.tid != tid) {
            return ErrCode::kTypeId;
        }
        data = found.ud;
        return ErrCode::kGood;
    }

    ErrCode removeBuf_i(const char* key) override
    {
        DS_ASSERT(key);
        std::string name(key);
        LockMutex locker(_mutex);
        auto iter = _hashTable.find(name);
        if (iter == _hashTable.end()) {
            return ErrCode::kGood;
        }
        removeDataInfo(iter->second);
        _hashTable.erase(iter);
        return ErrCode::kGood;
    }

    ErrCode clear_i() override
    {
        LockMutex locker(_mutex);
        removeAll();
        return ErrCode::kGood;
    }

    ErrCode copy_i(abiDataMap* input, DataMapPolicy policy, char* policyData)
    {
        return copyInput(input, policy, policyData);
    }

    ErrCode copy_i(abiDataMap* input, const char* key, DataMapPolicy policy, char* policyData)
    {
        return copyKey(input, policy, policyData, key);
    }

    std::string applyPolicy(DataMapPolicy policy, char* policyData, std::string keyName)
    {
        std::string newKeyName = keyName;
        if ((policy == DataMapPolicy::kCopyWithIntegerId) && policyData) {
            /** When copying, the key name need to change to keyName+%d */
            char* id = policyData;
            newKeyName = keyName + std::string("+") + std::string(id);
        }
        return newKeyName;
    }

    ErrCode copyInput(abiDataMap* input, DataMapPolicy policy, char* policyData)
    {
        DataHashTable* ht = (DataHashTable*)input;
        DS_ASSERT(ht);
        LockMutex lockerFrom(ht->_mutex);
        auto copiedMap = ht->_hashTable;
        ErrCode c = ErrCode::kGood;
        for (auto& i : copiedMap) {
            i.second.ud = i.second.ud->refCopy();
        }
        lockerFrom.unlock();

        LockMutex locker(_mutex);
        for (auto& iter : copiedMap) {
            std::string newKeyName = applyPolicy(policy, policyData, iter.first);
            c = setBufUnsafe(newKeyName.c_str(), iter.second.tid, iter.second.ud);
        }
        return c;
    }

    ErrCode copyKey(abiDataMap* input, DataMapPolicy policy, char* policyData, const char* key)
    {
        DataHashTable* ht = (DataHashTable*)input;
        DS_ASSERT(ht);
        std::string name(key);
        LockMutex lockerFrom(ht->_mutex);
        auto iter = ht->_hashTable.find(name);
        if (iter == _hashTable.end()) {
            return ErrCode::kNotFound;
        }
        auto tid = iter->second.tid;
        auto value = iter->second.ud->refCopy();
        lockerFrom.unlock();

        LockMutex locker(_mutex);
        std::string newKeyName = applyPolicy(policy, policyData, name);
        return setBufUnsafe(newKeyName.c_str(), tid, value);
    }

    int32_t getSize_i()
    {
        LockMutex locker(_mutex);
        return _hashTable.size();
    }

    void printDebug_i() const
    {
        if (!ENABLE_DEBUG) {
            return;
        }
        LockMutex locker(_mutex);
        for (const auto& each : _hashTable) {
            LOG_DEBUG("hash table entry = [%s]\n", each.first.c_str());
        }
    }

private:
    void removeAll()
    {
        for (auto each : _hashTable) {
            removeDataInfo(each.second);
        }
        _hashTable.clear();
    }

    void removeDataInfo(DataInfo& info)
    {
        if (info.ud) {
            info.ud->destroy();
        }
        info.ud = nullptr;
    }
    std::unordered_map<KeyName, DataInfo> _hashTable;
    mutable std::mutex _mutex;
};

}}  // namespace ds3d::impl

using namespace ds3d;

DS3D_EXTERN_C_BEGIN
DS3D_EXPORT_API abiRefDataMap*
NvDs3d_CreateDataHashMap()
{
    std::shared_ptr<abiDataMap> dataMap = std::make_shared<impl::DataHashTable>();
    return PtrToAbiRef<abiDataMap>(std::move(dataMap));
}
DS3D_EXTERN_C_END

#endif  // _DS3D_COMMON_IMPL_DATAHASHMAP__H
