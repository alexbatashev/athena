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

#include <athena/core/Node.h>

#include <vector>

namespace athena::core {

Node::Node(Node &&src) noexcept
    : AbstractNode(std::move(src)),
      mIncomingNodes(std::move(src.mIncomingNodes)),
      mOperation(src.mOperation) {}

Node::Node(Operation &&op)
    : AbstractNode(op.getName() + std::to_string(++mNodeCounter),
                   NodeType::DEFAULT),
      mOperation(op) {}

Node &Node::operator=(Node &&src) noexcept {
    mIncomingNodes = std::move(src.mIncomingNodes);
    mOutgoingNodes = std::move(src.mOutgoingNodes);
    mOperation     = std::move(src.mOperation);
    mName          = std::move(src.mName);
    return *this;
}

void Node::after(AbstractNode *node) {
    node->addOutgoingNode(node);
    mIncomingNodes.emplace_back(node);
}

const Operation &Node::getAssignedOperation() { return mOperation; }

NodeType Node::getType() { return NodeType::DEFAULT; }

}  // namespace athena::core