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

#include "SYCLEvent.h"
#include "SYCLDevice.h"

#include <utility>

namespace athena::backend::llvm {
SYCLEvent::SYCLEvent(SYCLDevice* device, cl::sycl::event evt)
    : mDevice(device), mEvent(std::move(evt)) {
  mFuture = std::async([&]() {
    mEvent.wait();
    for (auto& callback : mCallbacks) {
      callback();
    }
  });
};
void SYCLEvent::wait() { mEvent.wait(); }
auto SYCLEvent::getDevice() -> Device* { return mDevice; };
} // namespace athena::backend::llvm

