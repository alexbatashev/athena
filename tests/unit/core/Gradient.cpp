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

#include <polarai/core/context/Context.hpp>
#include <polarai/core/graph/Graph.hpp>
#include <polarai/core/node/InputNode.hpp>
#include <polarai/core/node/Node.hpp>
#include <polarai/core/node/OutputNode.hpp>
#include <polarai/operation/AddOperation.hpp>
#include <traversal/ContentChecker.hpp>
#include <traversal/EdgesChecker.hpp>
#include <traversal/TopologyChecker.hpp>

#include <gtest/gtest.h>

using namespace polarai;
using namespace polarai::core;
using namespace polarai::operation;
using namespace polarai::tests::unit;

namespace {
TEST(Gradient, Topology1) {
  Context context;
  auto graph = context.create<Graph>();
  auto inp1 =
      graph.create<InputNode>(TensorShape{2, 2}, DataType::FLOAT, false, 0);
  auto inp2 =
      graph.create<InputNode>(TensorShape{2, 2}, DataType::FLOAT, false, 0);
  auto operationId = context.create<AddOperation>();
  auto node = graph.create<Node>(operationId);
  graph.connect(inp1, node, AddOperation::LEFT);
  graph.connect(inp2, node, AddOperation::RIGHT);
  auto out = graph.create<OutputNode>();
//  graph.connect(node, out, Operation::Unmarked);
//   auto& traversal = graph.traverse();
//   auto [gradientGraph, connectGraph] = graph.getGradient(node);
}
} // namespace
