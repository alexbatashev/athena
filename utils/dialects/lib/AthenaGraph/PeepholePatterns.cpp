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

#include "AthenaGraph/AthenaGraphOps.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {
struct DoubleTransposePattern
    : public OpRewritePattern<ath_graph::TransposeOp> {
  DoubleTransposePattern(mlir::MLIRContext* context)
      : OpRewritePattern<ath_graph::TransposeOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(ath_graph::TransposeOp op,
                  mlir::PatternRewriter& rewriter) const override {
    // Look through the input of the current transpose.
    mlir::Value transposeInput = op.getOperand(0);
    ath_graph::TransposeOp transposeInputOp =
        llvm::dyn_cast<ath_graph::TransposeOp>(transposeInput.getDefiningOp());

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp)
      return failure();

    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOp(op, {transposeInputOp.getOperand(0)});
    return success();
  }
};
} // namespace

void ath_graph::TransposeOp::getCanonicalizationPatterns(
    OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<DoubleTransposePattern>(context);
}
