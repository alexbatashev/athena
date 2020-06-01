#include <athena/backend/llvm/LLVMExecutor.h>
#include <athena/backend/llvm/runtime/GraphHandle.h>

#include <gtest/gtest.h>

using namespace athena::backend::llvm;

constexpr static auto IR = R"(
module {
  "ath_graph.node"() ( {
    %0 = "ath_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<3xf32>
    "ath_graph.alloc"(%0) : (tensor<3xf32>) -> ()
    "ath_graph.lock"(%0) {lock_type = "read_write"} : (tensor<3xf32>) -> ()
    %1 = constant 42.0 : f32
    %2 = "ath_graph.fill"(%1, %0) : (f32, tensor<3xf32>) -> (tensor<3xf32>)
    "ath_graph.release"(%0) : (tensor<3xf32>) -> ()
    "ath_graph.return"(%2) : (tensor<3xf32>) -> ()
  }) {cluster_id = 0 : index, node_id = 0 : index, sym_name = "testNode", type = () -> tensor<3xf32>} : () -> ()
  "ath_graph.graph"() ( {
    %0 = "ath_graph.eval"() {node = @testNode} : () -> tensor<3xf32>
    "ath_graph.barrier"() {clusterId = 0 : index} : () -> ()
    "ath_graph.graph_terminator"() : () -> ()
  }) {sym_name = "testGraph", type = () -> ()} : () -> ()
}
)";

TEST(JITIntegration, FillOperationSample) {
  LLVMExecutor executor;
  executor.addModule(IR);

  GraphHandle handle;
  handle.allocator = executor.getAllocatorPtr();
  handle.devices = executor.getDevices();

  executor.execute("testGraph", &handle);
}