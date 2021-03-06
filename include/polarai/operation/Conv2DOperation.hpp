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

#include <polar_operation_export.h>
#include <polarai/core/operation/Operation.hpp>
#include <polarai/operation/internal/Conv2DOperationInternal.hpp>

namespace polarai::operation {
class POLAR_OPERATION_EXPORT Conv2DOperation : public core::Operation {
public:
  using InternalType = internal::Conv2DOperationInternal;
  enum Arguments { INPUT = 110, KERNEL };
};
} // namespace polarai::operation
