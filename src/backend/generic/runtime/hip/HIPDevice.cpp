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

#include "HIPDevice.hpp"

namespace polarai::backend::generic {
HIPDevice::HIPDevice(hipDevice_t device) : mDevice(device) {
  char name[100];
  hipError_t err = hipDeviceGetName(name, 100, mDevice);
  if (err != hipSuccess) {
    std::terminate(); // todo proper error handling
  }
  mDeviceName = std::string(name);
}

std::string HIPDevice::getDeviceName() const {
  return mDeviceName;
}
}
