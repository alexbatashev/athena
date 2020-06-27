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

#include <athena/backend/llvm/LLVMExecutor.h>
#include <athena/backend/llvm/runtime/GraphHandle.h>

#include <gtest/gtest.h>

using namespace athena::backend::llvm;

constexpr static auto IR = R"(
module {
  "ath_graph.node"() ( {
    %0 = "ath_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<32xf32>
    "ath_graph.alloc"(%0) : (tensor<32xf32>) -> ()
    "ath_graph.lock"(%0) {lock_type = "read_write"} : (tensor<32xf32>) -> ()
    %1 = constant 42.0 : f32
    %2 = "ath_graph.fill"(%1, %0) : (f32, tensor<32xf32>) -> (tensor<32xf32>)
    "ath_graph.release"(%0) : (tensor<32xf32>) -> ()
    "ath_graph.return"(%2) : (tensor<32xf32>) -> ()
  }) {cluster_id = 0 : index, node_id = 0 : index, sym_name = "testNode", type = () -> tensor<3xf32>} : () -> ()
  "ath_graph.graph"() ( {
    %0 = "ath_graph.eval"() {node = @testNode} : () -> tensor<32xf32>
    "ath_graph.barrier"() {clusterId = 0 : index} : () -> ()
    "ath_graph.graph_terminator"() : () -> ()
  }) {sym_name = "testGraph", type = () -> ()} : () -> ()
}
)";

TEST(JITIntegration, DISABLED_FillOperationSample) {
  LLVMExecutor executor;
  executor.addModule(IR);

  GraphHandle handle;
  handle.allocator = executor.getAllocatorPtr();
  handle.devices = executor.getDevices();

  executor.execute("testGraph", &handle);

  auto& allocator = executor.getAllocator();

  MemoryRecord record;
  record.virtualAddress = 1;
  record.allocationSize = 32 * 4;
  allocator.lock(record, LockType::READ);
  auto data = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < 32; i++) {
    EXPECT_FLOAT_EQ(data[i], 42.0f);
  }
  allocator.release(record);
}
