namespace DebuggerCore
{
    public interface ISystemDrawer
    {
        public void Scale(double width, double height);
        public void Rectangle(ulong id, double x, double y, double width, double height);
        public void Bezier(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4);

        public void Draw();
    }
}