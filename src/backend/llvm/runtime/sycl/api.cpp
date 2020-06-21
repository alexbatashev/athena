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

#include <athena/backend/llvm/runtime/api.h>
#include <athena/backend/llvm/runtime/runtime_export.h>

using namespace athena::backend::llvm;
using namespace cl::sycl;

extern "C" {
ATH_RT_LLVM_EXPORT DeviceContainer getAvailableDevices() {
  auto allDevices = device::get_devices(cl::sycl::info::device_type::host);
  auto* syclDevices = new Device*[allDevices.size()];

  #ifdef USES_COMPUTECPP
  if (allDevices.size() > 1) {
    allDevices.erase(allDevices.begin()); // remove host device
  }
  #endif

  int i = 0;
  for (const auto& device : allDevices) {
    syclDevices[i++] = new SYCLDevice(device);
  }

  DeviceContainer deviceContainer{syclDevices, allDevices.size()};
  return deviceContainer;
}
ATH_RT_LLVM_EXPORT void consumeDevice(Device* dev) { delete dev; }

ATH_RT_LLVM_EXPORT void consumeContainer(DeviceContainer cont) {
  delete[] cont.devices;
}
}
