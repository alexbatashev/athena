/*
 * Copyright (c) 2018 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#ifndef ATHENA_GRAPH_H
#define ATHENA_GRAPH_H

#include <athena/core/Optimizer.h>
#include <athena/core/Traversal.h>
#include <athena/core/core_export.h>
#include <athena/core/inner/Settings.h>
#include <athena/core/inner/Table.h>

#include <ostream>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace athena::core {
namespace inner {
struct ATH_CORE_EXPORT Edge {
  size_t startNodeIndex;
  size_t endNodeIndex;
  EdgeMark mark;
  Edge(size_t startNodeIndex, size_t endNodeIndex, EdgeMark mark)
      : startNodeIndex(startNodeIndex), endNodeIndex(endNodeIndex), mark(mark) {
  }
  bool operator==(const Edge& rhs) const {
    return startNodeIndex == rhs.startNodeIndex &&
           endNodeIndex == rhs.endNodeIndex;
  }
  bool operator<(const Edge& rhs) const {
    return startNodeIndex < rhs.startNodeIndex;
  }
};
} // namespace inner

using SyncStorage = std::unordered_set<size_t>;
using OwningStorage = inner::TupleContainers<std::vector, Node, InputNode,
                                             OutputNode, LossNode>::Holder;
using Topology = std::vector<inner::Edge>;

/**
 * A computation graph is an abstraction to represent an arbitrary function
 * in a way that is suitable for computation.
 */
class ATH_CORE_EXPORT Graph {
private:
  SyncStorage mSyncStorage;
  OwningStorage mOwningStorage;
  Topology mTopology;
  Context* mContext;
  size_t mGraphIndex;
  Traversal mTraversal;
  std::unique_ptr<Optimizer> mOptimizer;

  const std::string mGraphName;

  template <typename TemplateNodeType>
  void saveRealNode(TemplateNodeType& node, bool isRepairedNode, bool isErase);
  void saveNode(AbstractNode& node, bool isRepairedNode, bool isErase);
  ATHENA_REINITIALIZE void fullClear();

  void setUpTensors() const;

public:
  explicit Graph(Context& context);
  Graph(const Graph& rhs) = delete;
  Graph(Graph&& rhs) noexcept;
  ~Graph();

  Graph& operator=(const Graph& rhs) = delete;
  Graph& operator=(Graph&& rhs) = delete;

  const SyncStorage& getSyncStorage() const;
  const OwningStorage& getOwningStorage() const;
  const Topology& getTopology() const;

  /// Creates a node inside a Graph.
  ///
  /// \tparam NodeT is a type of node to be created.
  /// \tparam Args is a type of node constructor arguments.
  /// \param args are constructor arguments.
  /// \return an identifier of a node inside the Context.
  template <typename NodeT, typename... Args> size_t create(Args&&... args) {
    std::get<std::vector<NodeT>>(mOwningStorage)
        .emplace_back(std::forward<Args&&>(args)...);
    auto node = std::get<std::vector<NodeT>>(mOwningStorage).back();
    inner::setGraphIndex(node, mGraphIndex);
    return node.getNodeIndex();
  }

  template <typename NodeT>
  std::optional<NodeT&> lookup(const std::string& nodeName) {
    auto storage = std::get<std::vector<NodeT>>(mOwningStorage);
    auto it =
        std::find_if(storage.begin(), storage.end(), [&](const NodeT& node) {
          return node.getName() == nodeName;
        });
    if (it != storage.end()) {
      return std::optional<NodeT&>(*it);
    }
    return std::nullopt;
  }

  template <typename NodeT>
  std::optional<std::reference_wrapper<NodeT>> lookup(const size_t nodeId) {
    auto storage = std::get<std::vector<NodeT>>(mOwningStorage);
    auto it =
        std::find_if(storage.begin(), storage.end(), [&](const NodeT& node) {
          return node.getNodeIndex() == nodeId;
        });
    if (it != storage.end()) {
      return std::optional<std::reference_wrapper<NodeT>>(*it);
    }
    return std::nullopt;
  }

  AbstractNode& lookup(const size_t nodeId) {
    // fixme remove node table for good, replace with typed lookup.
    return *inner::getNodeTable(*mContext)[nodeId];
  }

  void connect(const AbstractNode& from, const AbstractNode& to,
               EdgeMark mark) {
    // todo replace link with connect
    link(from, to, mark);
  }

  void connect(size_t fromNodeId, size_t toNodeId, EdgeMark mark) {
    auto& from = lookup(fromNodeId);
    auto& to = lookup(toNodeId);
    connect(from, to, mark);
  }

  /**
   * Add node to Graph
   * @param node A node to be added
   */
  void addNode(AbstractNode& node);
  void saveNode(AbstractNode& node, bool isRepairedNode = true);
  void saveAllSyncNodes(bool isRepairedNode = true);
  /**
   * Remove node from Graph
   * @param node Node to be removed
   */
  void removeNode(AbstractNode& node);
  /**
   * Add oriented edge between two nodes
   * @param startNode Start Node
   * @param endNode End Node
   * @param mark
   */
  void link(const AbstractNode& startNode, const AbstractNode& endNode,
            EdgeMark mark);
  size_t countOwningNodes() const;
  size_t countSyncNodes() const;
  size_t getGraphIndex() const;
  /**
   * Resets object to initial state
   */
  void clear();
  /**
   *
   * @return if traversal is still valid
   */
  bool isValidTraversal() const;
  /**
   * Traverse current Graph and save the results inside object
   * @return A reference to result traversal
   */
  const Traversal& traverse();

  /**
   * Get last traversal for given Graph
   * @param graph An instance of Graph
   * @return A reference to last traversal
   */
  friend Traversal& inner::getTraversal(Graph& graph);

  friend Context& inner::getContext(athena::core::Graph& graph);

  /**
   * Set up Graph optimizer
   * @tparam Opt Optimizer class
   * @tparam Args Optimizer arguments type
   * @param args Optimizer arguments
   */
  template <typename Opt, typename... Args> void setUpOptimizer(Args... args) {
    mOptimizer = std::make_unique<Opt>(args...);
  }

  std::unique_ptr<Optimizer>& getOptimizer() { return mOptimizer; }

  /**
   *
   * @return Current graph name
   */
  std::string getGraphName() { return mGraphName; };

  /// Constructs a new Graph that computes gradient of this Graph.
  // std::pair<Graph, std::vector<size_t>> gradient(const AbstractNode&
  // startNode);
};
} // namespace athena::core

#endif // ATHENA_GRAPH_H
