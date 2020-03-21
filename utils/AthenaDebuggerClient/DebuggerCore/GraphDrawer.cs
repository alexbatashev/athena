using ProtoGraph;
using Svg;
using Microsoft.Msagl.Core.Geometry.Curves;
using Rectangle = Microsoft.Msagl.Core.Geometry.Rectangle;

namespace DebuggerCore
{
    public class GraphDrawer<TSystemDrawer> where TSystemDrawer : ISystemDrawer
    {
        private TSystemDrawer _drawer;

        public GraphDrawer(TSystemDrawer drawer)
        {
            _drawer = drawer;
        }

        public void RenderGraph(Graph graph)
        {
            var drawableGraph = GraphLayoutGenerator.Generate(graph);

            //_drawer.Scale(drawableGraph.Width, drawableGraph.Height);

            foreach (var node in drawableGraph.Nodes)
            {
                _drawer.Rectangle(node.UserData is ulong data ? data : 0, node.BoundingBox.Left, node.BoundingBox.Top,
                    node.Width, node.Height);
            }

            foreach (var edge in drawableGraph.Edges)
            {
                if (!(edge.Curve is Curve c)) continue;
                foreach (var segment in c.Segments)
                {
                    if (segment is CubicBezierSegment bezierSegment)
                    {
                        _drawer.Bezier(bezierSegment.B(0).X, bezierSegment.B(1).Y, bezierSegment.B(1).X,
                            bezierSegment.B(1).Y, bezierSegment.B(2).X, bezierSegment.B(2).Y, bezierSegment.B(3).X,
                            bezierSegment.B(3).Y);
                    }
                }
            }
        }
    }
}