using Cairo;
using DebuggerCore;

namespace Frontend_GTK
{
    public class CairoDrawer : ISystemDrawer
    {
        private readonly Cairo.Context _cairoContext;

        public CairoDrawer(Cairo.Context ctx)
        {
            _cairoContext = ctx;
        }

        public void Scale(double width, double height)
        {
 //           _cairoContext.Scale(width, height);
//            _cairoContext.Scale(500, 480);
//            _cairoContext.Translate(0.5, 0.5);
            _cairoContext.Save();
            _cairoContext.SetSourceRGB(0.2, 0.2, 0.2);
            _cairoContext.Paint();
            _cairoContext.Restore();
        }

        public void Rectangle(ulong id, double x, double y, double width, double height)
        {
            System.Console.Out.Write("Rectangle " + x + ", " + y + ", " + width + ", " + height + "\n");
            _cairoContext.Rectangle(new PointD(0.5, 0.5), 0.3, 0.3);
            _cairoContext.Save();
            _cairoContext.SetSourceRGB(0.1, 1.0, 1.0);
            _cairoContext.Fill();
            //_cairoContext.Paint();
        }

        public void Bezier(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4)
        {
            _cairoContext.MoveTo(x1, y1);
            _cairoContext.CurveTo(x2, y2, x3, y3, x4, y4);
        }

        public void Draw()
        {
            _cairoContext.Paint();
        }
    }
}