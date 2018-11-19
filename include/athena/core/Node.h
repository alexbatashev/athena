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

#ifndef ATHENA_NODE_H
#define ATHENA_NODE_H

#include <vector>
#include <athena/core/Operation.h>

namespace athena::core {

/**
 * The class represents a single node of a computation graph. It has a name, a set of incoming nodes,
 * and a set of outgoing nodes. It also encapsulates Operation.
 */
class Node {
 protected:
    std::vector<Node*> mIncomingNodes;
    std::vector<Node*> mOutgoingNodes;
    Operation mOperation;
    std::string mName;
    static size_t mNodeCounter;

 public:
    Node() = delete;
    Node(const Node& src) = delete;
    Node(Node&& src) noexcept;
    explicit Node(Operation&& op);
    ~Node() = default;  //TODO rewrite if change Node* in vectors to shared_ptr<Node>. Now must deleted by Graph.
    Node& operator=(const Node& src) = delete;
    Node& operator=(Node&& src) noexcept;
    virtual void after(Node* node);
};

}

#endif //ATHENA_NODE_H
