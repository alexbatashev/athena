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

#include <polarai/core/graph/Traversal.hpp>
#include <polarai/utils/Index.hpp>

namespace polarai::tests::unit {
bool checkTraversalContent(
    const core::Traversal& traversal,
    const std::vector<std::set<polarai::utils::Index>>& target);
}
