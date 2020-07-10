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

namespace athena::backend::llvm {
class MetalContext : public Context {
public:
  MetalContext();
  std::vector<std::shared_ptr<Device>>& getDevices() override {
    return mDevices;
  };

private:
  std::vector<std::shared_ptr<Device>> mDevices;
};
} // namespace athena::backend::llvm