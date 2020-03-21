using System;
using System.Collections.Generic;
using ProtoGraph;
using Microsoft.Msagl.Core.Geometry.Curves;
using Microsoft.Msagl.Core.Layout;
using Microsoft.Msagl.Core.Routing;
using Microsoft.Msagl.Layout.Layered;
using Edge = Microsoft.Msagl.Core.Layout.Edge;
using DrawingNode = Microsoft.Msagl.Core.Layout.Node;

namespace DebuggerCore
{
    /// <summary>
    /// Helper class to generate nodes layout before drawing.
    /// </summary> 
    public class GraphLayoutGenerator
    {
        private const int FontSize = 13;
        private const int SidePadding = 16;
        private const int TopPadding = 8;
        private const int LineHeight = 16;
        private const int HorizontalNodeSpace = 20;
        private const int VerticalNodeSpace = 40;

        private static GeometryGraph ConvertGraph(Graph graph)
        {
            var drawableGraph = new GeometryGraph();
            var nodesCache = new Dictionary<ulong, DrawingNode>();

            AddNodeToDrawableGraph(graph, nodesCache, drawableGraph);

            foreach (var graphNode in graph.Nodes)
            {
                var newNode = new DrawingNode {UserData = graphNode.Index};
                nodesCache.Add(graphNode.Index, newNode);
                drawableGraph.Nodes.Add(newNode);
            }

            foreach (var lossNode in graph.LossNodes)
            {
                var newNode = new DrawingNode {UserData = lossNode.Index};
                nodesCache.Add(lossNode.Index, newNode);
                drawableGraph.Nodes.Add(newNode);
            }

            foreach (var outputNode in graph.OutputNodes)
            {
                var newNode = new DrawingNode {UserData = outputNode.Index};
                nodesCache.Add(outputNode.Index, newNode);
                drawableGraph.Nodes.Add(newNode);
            }

            foreach (var graphEdge in graph.Edges)
            {
                var newEdge = new Edge(nodesCache[graphEdge.Start], nodesCache[graphEdge.End]);
                drawableGraph.Edges.Add(newEdge);
            }

            return drawableGraph;
        }

        private static void AddNodeToDrawableGraph(Graph graph, IDictionary<ulong, DrawingNode> nodesCache,
            GeometryGraph drawableGraph)
        {
            foreach (var inputNode in graph.InputNodes)
            {
                var newNode = new DrawingNode {UserData = inputNode.Index};
                nodesCache.Add(inputNode.Index, newNode);
                drawableGraph.Nodes.Add(newNode);
            }
        }

        public static GeometryGraph Generate(Graph graph)
        {
            var drawableGraph = ConvertGraph(graph);

            var settings = new SugiyamaLayoutSettings {
                Transformation = PlaneTransformation.Rotation(Math.PI/2),
                EdgeRoutingSettings = {EdgeRoutingMode = EdgeRoutingMode.Spline}
            };
            
            var layout = new LayeredLayout(drawableGraph, settings);
            layout.Run();

            return drawableGraph;
        }
    }
}