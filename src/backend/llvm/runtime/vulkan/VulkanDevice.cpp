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

#include "VulkanDevice.h"

#include <vulkan/vulkan.h>

static uint32_t getComputeQueueFamilyIndex(VkPhysicalDevice physicalDevice) {
  uint32_t queueFamilyCount;

  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                           NULL);

  // Retrieve all queue families.
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                           queueFamilies.data());

  // Now find a family that supports compute.
  uint32_t i = 0;
  for (; i < queueFamilies.size(); ++i) {
    VkQueueFamilyProperties props = queueFamilies[i];

    if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
      // found a queue with compute. We're done!
      break;
    }
  }

  if (i == queueFamilies.size()) {
    std::terminate(); // todo no queue families with compute support, throw.
  }

  return i;
}

namespace athena::backend::llvm {
VulkanDevice::VulkanDevice(VkPhysicalDevice device) : mPhysicalDevice(device) {

  VkDeviceQueueCreateInfo queueCreateInfo = {};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  uint32_t queueFamilyIndex = getComputeQueueFamilyIndex(mPhysicalDevice);
  queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
  queueCreateInfo.queueCount = 1;
  float queuePriorities = 1.0;
  queueCreateInfo.pQueuePriorities = &queuePriorities;

  VkDeviceCreateInfo deviceCreateInfo = {};

  VkPhysicalDeviceFeatures deviceFeatures = {};

  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.enabledLayerCount = 0;
  deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
  deviceCreateInfo.queueCreateInfoCount = 1;
  deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

  vkCreateDevice(mPhysicalDevice, &deviceCreateInfo, nullptr, &mDevice);
}

std::string VulkanDevice::getDeviceName() const { return mDeviceName; }
} // namespace athena::backend::llvm
