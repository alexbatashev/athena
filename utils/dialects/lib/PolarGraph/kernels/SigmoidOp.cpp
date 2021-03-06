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

#include "PolarGraph/PolarGraphOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"

namespace mlir::polar_graph {
void SigmoidOp::produceKernel(OpBuilder& builder,
                              Block::BlockArgListType args) {
  auto memrefTy = args.back().getType().cast<MemRefType>();
  auto tensorTy = out().getType().cast<RankedTensorType>();
  auto zero = builder.create<ConstantIndexOp>(builder.getUnknownLoc(), 0);

  SmallVector<Value, 3> lbs(memrefTy.getRank(), zero);
  SmallVector<Value, 3> ubs;
  SmallVector<int64_t, 3> steps(memrefTy.getRank(), 1);

  for (int i = 0; i < memrefTy.getRank(); i++) {
    auto dim = builder.create<ConstantIndexOp>(builder.getUnknownLoc(),
                                               tensorTy.getDimSize(i));
    ubs.push_back(dim);
  }

  auto bodyBuilder = [args, memrefTy](OpBuilder& builder, Location loc,
                                      ValueRange idx) {
    auto in =
        builder.create<AffineLoadOp>(builder.getUnknownLoc(), args[0], idx);
    auto negIn = builder.create<NegFOp>(builder.getUnknownLoc(), in);
    auto exp = builder.create<ExpOp>(builder.getUnknownLoc(), negIn);
    auto one = builder.create<ConstantFloatOp>(
        builder.getUnknownLoc(), APFloat(1.0f),
        memrefTy.getElementType().cast<FloatType>());
    auto sum = builder.create<AddFOp>(builder.getUnknownLoc(), one, exp);

    auto res = builder.create<DivFOp>(builder.getUnknownLoc(), one, sum);

    builder.create<AffineStoreOp>(loc, res, args[1], idx);
  };
  buildAffineLoopNest(builder, builder.getUnknownLoc(), lbs, ubs, steps,
                      bodyBuilder);
}
} // namespace mlir::polar_graph
