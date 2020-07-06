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

#pragma once

#include <athena/backend/llvm/runtime/Context.h>

#include <volk.h>

namespace athena::backend::llvm {
class VulkanContext : public Context {
public:
  VulkanContext(VkInstance instance);
  std::vector<std::shared_ptr<Device>>& getDevices() override;

private:
  std::vector<std::shared_ptr<Device>> mDevices;
  VkInstance mInstance; 
};
} // namespace athena::backend::llvm