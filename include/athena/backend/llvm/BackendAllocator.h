//===----------------------------------------------------------------------===//
// Copyright (c) 2020 Athena. All rights reserved.
// https://getathena.ml
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#ifndef ATHENA_BACKENDALLOCATOR_H
#define ATHENA_BACKENDALLOCATOR_H

#include <athena/backend/llvm/runtime/Device.h>
#include <athena/core/Allocator.h>

namespace athena::backend::llvm {
class BackendAllocator : public core::Allocator {
protected:
  virtual void* getImpl(const core::inner::Tensor& tensor, Device& device) = 0;

public:
  virtual ~BackendAllocator() = default;

  virtual void registerDevice(Device &device) = 0;

  virtual void allocate(const core::inner::Tensor& tensor, Device& device) = 0;

  virtual void lock(const core::inner::Tensor& tensor, Device& device) = 0;

  template <typename BufferT>
  BufferT* get(const core::inner::Tensor& tensor, Device& device) {
    return reinterpret_cast<BufferT*>(getImpl(tensor, device));
  }
};
} // namespace athena::backend::llvm

#endif // ATHENA_BACKENDALLOCATOR_H
