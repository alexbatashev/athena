using System;
using System.Collections.Generic;
using System.Linq;
using ProtoGraph;

namespace DebuggerCore
{
    /// <summary>
    /// Helper class to generate nodes layout before drawing.
    /// </summary> 
    public class LayoutGenerator
    {
        private const int FontSize = 13;
        private const int SidePadding = 16;
        private const int TopPadding = 8;
        private const int LineHeight = 16;
        private const int HorizontalNodeSpace = 20;
        private const int VerticalNodeSpace = 40;

        private class NodeHolder
        {
            private ProtoGraph.Node Node;
            private ProtoGraph.InputNode InputNode;
            private ProtoGraph.LossNode LossNode;
            private ProtoGraph.OutputNode OutputNode;

            private enum NodeType
            {
                Node,
                Input,
                Loss,
                Output
            }

            private readonly NodeType ContainedNodeType;

            public NodeHolder(ProtoGraph.Node node)
            {
                Node = node;
                ContainedNodeType = NodeType.Node;
            }

            public NodeHolder(ProtoGraph.InputNode node)
            {
                InputNode = node;
                ContainedNodeType = NodeType.Input;
            }

            public NodeHolder(ProtoGraph.LossNode node)
            {
                LossNode = node;
                ContainedNodeType = NodeType.Loss;
            }

            public NodeHolder(ProtoGraph.OutputNode node)
            {
                OutputNode = node;
                ContainedNodeType = NodeType.Output;
            }

            public string GetName()
            {
                return ContainedNodeType switch
                {
                    NodeType.Input => InputNode.Name,
                    NodeType.Node => Node.Name,
                    NodeType.Loss => LossNode.Name,
                    NodeType.Output => OutputNode.Name,
                    _ => throw new ArgumentOutOfRangeException()
                };
            }

            public ulong GetId()
            {
                return ContainedNodeType switch
                {
                    NodeType.Input => InputNode.Index,
                    NodeType.Node => Node.Index,
                    NodeType.Loss => LossNode.Index,
                    NodeType.Output => OutputNode.Index,
                    _ => throw new ArgumentOutOfRangeException()
                };
            }
        }

        private static Dictionary<ulong, List<ulong>> CreateStartEdgeIndex(Graph graph)
        {
            var index = new Dictionary<ulong, List<ulong>>();
            foreach (var edge in graph.Edges)
            {
                index[edge.Start].Add(edge.End);
            }

            return index;
        }

        private static Dictionary<ulong, List<ulong>> CreateEndEdgeIndex(Graph graph)
        {
            var index = new Dictionary<ulong, List<ulong>>();
            foreach (var edge in graph.Edges)
            {
                index[edge.End].Add(edge.Start);
            }

            return index;
        }

        private static List<NodeHolder> GetAllNodes(Graph graph)
        {
            var result = graph.Nodes.Select(node => new NodeHolder(node)).ToList();
            result.AddRange(graph.InputNodes.Select(node => new NodeHolder(node)));
            result.AddRange(graph.LossNodes.Select(node => new NodeHolder(node)));
            result.AddRange(graph.OutputNodes.Select(node => new NodeHolder(node)));

            return result;
        }

        public DrawableGraph Generate(Graph graph)
        {
            var clusters = new List<List<NodeHolder>>();

            var startIndex = CreateStartEdgeIndex(graph);
            var endIndex = CreateEndEdgeIndex(graph);
            var allNodes = GetAllNodes(graph);

            var toWatch = new Queue<NodeHolder>();
            var inputsCount = new Dictionary<ulong, ulong>();
            var clusterPlacement = new Dictionary<ulong, int>();

            foreach (var node in allNodes.Where(node => !startIndex.ContainsKey(node.GetId())))
            {
                toWatch.Append(node);
                inputsCount[node.GetId()] = 0;
                clusterPlacement[node.GetId()] = 0;
            }

            while (toWatch.Count > 0)
            {
                var node = toWatch.Dequeue();
                if (clusters.Count <= (int) inputsCount[node.GetId()])
                {
                    clusters.Add(new List<NodeHolder>());
                }

                clusters[clusterPlacement[node.GetId()]].Add(node);

                // traverse all children
                foreach (var child in allNodes.Where(curNode => startIndex[node.GetId()].Contains(curNode.GetId())))
                {
                    if (inputsCount.ContainsKey(child.GetId()))
                    {
                        inputsCount[child.GetId()] += 1;
                    }
                    else
                    {
                        inputsCount[child.GetId()] = 1;
                    }

                    if (inputsCount[child.GetId()] != (ulong) endIndex[child.GetId()].Count) continue;
                    toWatch.Append(child);
                    clusterPlacement[child.GetId()] = clusterPlacement[node.GetId()] + 1;
                }
            }

            var drawableGraph = new DrawableGraph();

            var lastX = 0;
            var lastY = 0;
            var isFirstCluster = true;
            var nodesCache = new Dictionary<ulong, DrawableNode>();

            foreach (var cluster in clusters)
            {
                foreach (var nodeHolder in cluster)
                {
                    lastX += HorizontalNodeSpace;

                    var width = nodeHolder.GetName().Length * FontSize + 2 * SidePadding;
                    const int height = 2 * FontSize + 2 * TopPadding;

                    var node = new DrawableNode(nodeHolder.GetName(), "", nodeHolder.GetId(), lastX, lastY, width,
                        height);
                    nodesCache[nodeHolder.GetId()] = node;

                    if (isFirstCluster)
                    {
                        drawableGraph.AddFirstLevelNode(node);
                    }
                    else
                    {
                        foreach (var parentIdx in endIndex[nodeHolder.GetId()])
                        {
                            nodesCache[parentIdx].AddChild(node);
                        }
                    }

                    lastX += width;
                }

                lastY += 2 * LineHeight + 2 * TopPadding + VerticalNodeSpace;
                isFirstCluster = false;
            }

            return drawableGraph;
        }
    }
}