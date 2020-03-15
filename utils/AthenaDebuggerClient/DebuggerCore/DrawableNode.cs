using System.Collections.Generic;

namespace DebuggerCore
{
    public class DrawableNode
    {
        private readonly List<DrawableNode> _children = new List<DrawableNode>();
        private string _name, _payload;
        private ulong _id;
        private int _x, _y, _width, _height;

        public DrawableNode(string name, string payload, ulong id, int x, int y, int width, int height)
        {
            _name = name;
            _payload = payload;
            _id = id;
            _x = x;
            _y = y;
            _width = width;
            _height = height;
        }

        public void AddChild(DrawableNode node)
        {
            _children.Add(node);
        }
    }
}