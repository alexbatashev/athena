using System;
using System.Collections.Generic;
using Microsoft.Msagl.Core.Geometry;
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

        private static int CalculateWidth(int charCount)
        {
            return SidePadding * 2 + FontSize * charCount;
        }

        private static int CalculateHeight()
        {
            return TopPadding * 2 + FontSize;
        }

        private static GeometryGraph ConvertGraph(Graph graph)
        {
            var drawableGraph = new GeometryGraph();
            var nodesCache = new Dictionary<ulong, DrawingNode>();

            foreach (var graphNode in graph.Nodes)
            {
                var newNode = new DrawingNode(
                    CurveFactory.CreateRectangle(40, 10, new Point()),
                    graphNode.Index);
                nodesCache.Add(graphNode.Index, newNode);
                drawableGraph.Nodes.Add(newNode);
            }

            foreach (var lossNode in graph.LossNodes)
            {
                var newNode = new DrawingNode(
                    CurveFactory.CreateRectangle(40, 10, new Point()),
                    lossNode.Index);
                nodesCache.Add(lossNode.Index, newNode);
                drawableGraph.Nodes.Add(newNode);
            }

            foreach (var outputNode in graph.OutputNodes)
            {
                var newNode = new DrawingNode(
                    CurveFactory.CreateRectangle(40, 10, new Point()),
                    outputNode.Index);
                nodesCache.Add(outputNode.Index, newNode);
                drawableGraph.Nodes.Add(newNode);
            }

            foreach (var inputNode in graph.InputNodes)
            {
                var newNode = new DrawingNode(
                    CurveFactory.CreateRectangle(40, 10, new Point()),
                    inputNode.Index);
                nodesCache.Add(inputNode.Index, newNode);
                drawableGraph.Nodes.Add(newNode);
                
            }

            foreach (var graphEdge in graph.Edges)
            {
                var newEdge = new Edge(nodesCache[graphEdge.Start], nodesCache[graphEdge.End]);
                drawableGraph.Edges.Add(newEdge);
            }

            return drawableGraph;
        }
        
        public static GeometryGraph Generate(Graph graph)
        {
            var drawableGraph = ConvertGraph(graph);

            var settings = new SugiyamaLayoutSettings
            {
                
                EdgeRoutingSettings = {EdgeRoutingMode = EdgeRoutingMode.Spline}
            };

            var layout = new LayeredLayout(drawableGraph, settings);
            layout.Run();

            return drawableGraph;
        }
    }
}