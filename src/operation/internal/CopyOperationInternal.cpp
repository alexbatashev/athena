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

#include <polarai/core/node/internal/AbstractNodeInternal.hpp>
#include <polarai/core/node/internal/NodeInternal.hpp>
#include <polarai/loaders/internal/ConstantLoaderInternal.hpp>
#include <polarai/operation/internal/CopyOperationInternal.hpp>

using namespace polarai::core::internal;

namespace polarai::operation::internal {
CopyOperationInternal::CopyOperationInternal(
    utils::WeakPtr<core::internal::ContextInternal> context,
    utils::Index publicNodeIndex, utils::String name)
    : OperationInternal(std::move(context), publicNodeIndex, std::move(name)) {}

utils::Index CopyOperationInternal::createResultTensor(
    utils::SharedPtr<core::internal::ContextInternal> context,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors) const {
  // TODO check preconditions
  auto dataType = tensors[0]->getDataType();
  auto tensorShape = tensors[0]->getShape();
  return context->create<core::internal::TensorInternal>(
      context, context->getNextPublicIndex(), dataType, std::move(tensorShape));
}

core::internal::GenValue CopyOperationInternal::gen(
    utils::SharedPtr<core::internal::ContextInternal> context,
    Generator& generator,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors,
    const core::internal::TensorInternal* resultTensor,
    GenNode parentNode) const {
  generator.setInsertionPoint(parentNode);

  std::unordered_map<utils::Index, GenValue> argMap;
  GenValue input = parentNode.getOperand(0);
  argMap[tensors.at(0)->getPublicIndex()] = input;

  std::unordered_map<utils::Index, GenValue> resultMap;
  GenValue out = parentNode.getResult();
  resultMap[resultTensor->getPublicIndex()] = out;

  lockTensors(generator, argMap, resultMap);

  generator.callBuiltin<builtin::Copy>(input, out);

  releaseTensors(generator, argMap, resultMap);
  return out;
}

std::tuple<utils::Index, std::vector<core::internal::Edge>,
           std::vector<utils::Index>>
CopyOperationInternal::genDerivative(
    const core::NodeState* inputNodeState,
    const core::NodeState* currentNodeState, size_t indexOfOutputDependence,
    utils::Index gradientGraphFinalNodeIndex) const {
  // TODO
  return {};
}

size_t CopyOperationInternal::getOperandsCount() const { return 1; }
} // namespace polarai::operation::internal
