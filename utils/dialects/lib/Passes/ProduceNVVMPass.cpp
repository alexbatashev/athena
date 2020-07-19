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

#include "Passes/Passes.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/NVVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/IR/Module.h"

using namespace mlir;

namespace {
class ProduceNVVMPass
    : public PassWrapper<ProduceNVVMPass, OperationPass<gpu::GPUModuleOp>> {
public:
  explicit ProduceNVVMPass(OptimizeModuleCallback callback)
      : mCallback(callback) {}

protected:
  void runOnOperation() override {
    auto module = getOperation();

    std::unique_ptr<llvm::Module> res = translateModuleToNVVMIR(module);
    if (sizeof(void*) == 8) {
      res->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                         "i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-"
                         "v64:64:64-v128:128:128-n16:32:64");
      res->setTargetTriple("nvptx64-unknown-cuda");
    } else {
      res->setDataLayout("e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                         "i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-"
                         "v64:64:64-v128:128:128-n16:32:64");
      res->setTargetTriple("nvptx-unknown-cuda");
    }
    mCallback(res);

    std::string out;
    llvm::raw_string_ostream stream(out);
    stream << *res;
    stream.flush();

    module.setAttr("nvvm", StringAttr::get(out, &getContext()));

    module.getBody()->clear();
    OpBuilder builder(&getContext());
    builder.setInsertionPointToStart(module.getBody());
    builder.create<gpu::ModuleEndOp>(builder.getUnknownLoc());
  }

private:
  OptimizeModuleCallback mCallback;
};
} // namespace

namespace mlir {
auto createProduceNVVMModulePass(OptimizeModuleCallback callback)
    -> std::unique_ptr<OperationPass<gpu::GPUModuleOp>> {
  return std::make_unique<ProduceNVVMPass>(callback);
}
} // namespace mlir
