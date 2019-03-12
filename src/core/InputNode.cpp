/*
 * Copyright (c) 2018 Athena. All rights reserved.
 * https://athenaframework.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include "athena/core/InputNode.h"

#include <athena/core/InputNode.h>

namespace athena::core {

InputNode::InputNode(InputNode &&node) noexcept
    : AbstractNode(std::move(node)), mTensor(node.mTensor) {
    node.mTensor = nullptr;
}

InputNode &InputNode::operator=(InputNode &&src) noexcept {
    mOutgoingNodes = std::move(src.mOutgoingNodes);
    mTensor        = src.mTensor;
    mName          = std::move(src.mName);
    src.mTensor    = nullptr;
    return *this;
}

void InputNode::after(AbstractNode *node) {
    FatalError("Error. Input node can not be after something!");
}

InputNode::InputNode(Tensor *tensor)
    : AbstractNode("Input" + std::to_string(++mNodeCounter), NodeType::INPUT),
      mTensor(tensor) {}

Tensor *InputNode::getData() { return mTensor; }

NodeType InputNode::getType() { return NodeType::INPUT; }

}  // namespace athena::core