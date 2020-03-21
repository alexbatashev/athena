using System;
using Cairo;
using DebuggerCore;
using Gtk;
using ProtoGraph;
using UI = Gtk.Builder.ObjectAttribute;

namespace Frontend_GTK
{
    internal class MainWindow : Window
    {
        [UI] private readonly DrawingArea _graphDisplay = null;

        private int _counter;

        public MainWindow() : this(new Builder("MainWindow.glade"))
        {
        }

        private MainWindow(Builder builder) : base(builder.GetObject("MainWindow").Handle)
        {
            builder.Autoconnect(this);

            DeleteEvent += Window_DeleteEvent;
            _graphDisplay.Drawn += new DrawnHandler(DrawnCallback);
        }

        private void DrawnCallback(object o, DrawnArgs args)
        {
            var cairoContext = args.Cr;
            cairoContext.GetTarget().Dispose();
            var cairoDrawer = new CairoDrawer(cairoContext);
            cairoDrawer.Scale(_graphDisplay.AllocatedWidth, _graphDisplay.AllocatedHeight);

            var g = new Graph();
            var inpNode = new InputNode
            {
                Name = "InpTest",
                Index = 0,
                IsFrozen = false,
                Loader = null,
                Tensor = null
            };
            var node1 = new ProtoGraph.Node
            {
                Name = "Test",
                Index = 1,
                InputsCount = 1,
                Operation = null,
                Tensor = null
            };
            g.Nodes.Add(node1);
            g.InputNodes.Add(inpNode);

            var edge = new Edge
            {
                End = 1,
                Mark = 0,
                Start = 0
            };
            
            g.Edges.Add(edge);
            
            var graphDrawer = new GraphDrawer<CairoDrawer>(cairoDrawer);
            graphDrawer.RenderGraph(g);
        }

        private void Window_DeleteEvent(object sender, DeleteEventArgs a)
        {
            Application.Quit();
        }
    }
}