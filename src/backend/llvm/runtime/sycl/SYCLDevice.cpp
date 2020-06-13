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

#include "SYCLDevice.h"
#include "../utils/utils.h"
#include "SYCLEvent.h"

#include <athena/backend/llvm/BackendAllocator.h>
#include <athena/backend/llvm/runtime/LaunchCommand.h>
#include <athena/backend/llvm/runtime/TensorInfo.h>

#include "kernels/AddNaive.hpp"
#include "kernels/Fill.hpp"

using namespace cl::sycl;

namespace athena::backend::llvm {
auto SYCLDevice::launch(BackendAllocator& allocator, LaunchCommand& cmd,
                        Event* dependency) -> Event* {
  if (dependency) {
    // fixme there's no clear way in SYCL to launch kernel after the event is
    //  completed. But due to the barrier syncs this most likely will be a
    //  non-blocking call. In future runtimes will need a unified event system.
    dependency->wait();
  }
  if (mKernelMap.count(cmd.kernelName)) {
    return mKernelMap.at(cmd.kernelName)(this, allocator, cmd, dependency);
  }

  // todo implement interoperability kernel launch
  return nullptr;
}

void SYCLDevice::addModule(ProgramDesc) {}
void SYCLDevice::linkModules() {}

void SYCLDevice::consumeEvent(Event* evt) {
  evt->wait();
  delete evt;
}

void SYCLDevice::populateKernelMap() {
  if (mUsesUSM) {
    mKernelMap["ffill"] = FillWrapper<AllocatorType::usm, float>{};
    mKernelMap["fadd"] = AddKernelWrapper<AllocatorType::usm, float>{};
  }
}
} // namespace athena::backend::llvm
