using ProtoGraph;
using Svg;
using Microsoft.Msagl.Core.Geometry.Curves;
using Rectangle = Microsoft.Msagl.Core.Geometry.Rectangle;

namespace DebuggerCore
{
    public class GraphDrawer<SystemDrawer> where SystemDrawer : ISystemDrawer
    {
        private SystemDrawer _drawer;

        public GraphDrawer(SystemDrawer drawer)
        {
            _drawer = drawer;
        }
        
        public string RenderGraph(Graph graph)
        {
            var svgDoc = new SvgDocument();
            var drawableGraph = GraphLayoutGenerator.Generate(graph);

            svgDoc.Width = (SvgUnit) drawableGraph.Width;
            svgDoc.Height = (SvgUnit) drawableGraph.Height;

            foreach (var node in drawableGraph.Nodes)
            {
                var rectangle = new SvgRectangle
                {
                    Width = (SvgUnit) node.Width,
                    Height = (SvgUnit) node.Height,
                    X = (SvgUnit) node.BoundingBox.Left,
                    Y = (SvgUnit) node.BoundingBox.Top
                };

                svgDoc.Children.Add(rectangle);
            }

            foreach (var edge in drawableGraph.Edges)
            {
                var line = new SvgLine
                {
                    StartX = (SvgUnit) edge.Curve.Start.X,
                    StartY = (SvgUnit) edge.Curve.Start.Y,
                    EndX = (SvgUnit) edge.Curve.End.X,
                    EndY = (SvgUnit) edge.Curve.End.Y
                };
                svgDoc.Children.Add(line);
            }
            
            return "";
        } 
    }
}