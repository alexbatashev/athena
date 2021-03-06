//===----------------------------------------------------------------------===//
// Copyright (c) 2020 PolarAI. All rights reserved.
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#pragma once

#include "utils.hpp"
#include <polarai/backend/generic/AllocatorLayerBase.hpp>
#include <polarai/backend/generic/MemoryRecord.hpp>
#include <polarai/utils/error/FatalError.hpp>

#include <cuda.h>

#include <unordered_map>
#include <unordered_set>

namespace polarai::backend::generic {
class CudaAllocator : public AllocatorLayerBase {
private:
  MemoryOffloadCallbackT mOffloadCallback;
  std::unordered_map<MemoryRecord, CUdeviceptr> mMemMap;
  std::unordered_set<MemoryRecord> mLockedAllocations;
  std::unordered_set<MemoryRecord> mReleasedAllocations;
  std::unordered_map<MemoryRecord, int> mTags;

  void freeMemory(MemoryRecord record) {
    size_t freedMem = 0;
    while (freedMem < record.allocationSize) {
      if (mReleasedAllocations.size() == 0)
        std::terminate();
      MemoryRecord alloc = *mReleasedAllocations.begin();
      freedMem += alloc.allocationSize;
      mOffloadCallback(alloc, *this);
      check(cuMemFree(mMemMap[alloc]));
      mMemMap.erase(alloc);
      mReleasedAllocations.erase(alloc);
    }
  }

public:
  CudaAllocator() = default;
  ~CudaAllocator() override = default;

  void registerMemoryOffloadCallback(MemoryOffloadCallbackT function) override {
  }
  void allocate(MemoryRecord record) override {
    if (mMemMap.count(record))
      return; // no double allocations are allowed

    CUdeviceptr mem;
    check(cuMemAlloc(&mem, record.allocationSize));

    mMemMap[record] = mem;
    mTags[record] = 1;
  }
  void deallocate(MemoryRecord record) override {
    if (mLockedAllocations.count(record)) {
      std::terminate();
    }

    delete[] reinterpret_cast<unsigned char*>(mMemMap[record]);

    if (mReleasedAllocations.count(record)) {
      mReleasedAllocations.erase(record);
    }
    mTags[record] = 0;
  }
  void lock(MemoryRecord record) override { mLockedAllocations.insert(record); }
  void release(MemoryRecord record) override {
    mLockedAllocations.erase(record);
    mReleasedAllocations.insert(record);
  }

  void* getPtr(MemoryRecord record) override { return &mMemMap[record]; }

  bool isAllocated(const MemoryRecord& record) const override {
    return mMemMap.count(record) > 0;
  }

  size_t getTag(MemoryRecord record) override { return mTags[record]; }

  void setTag(MemoryRecord record, size_t tag) override { mTags[record] = tag; }
};
} // namespace polarai::backend::generic
