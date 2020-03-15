using System.Collections.Generic;

namespace DebuggerCore
{
    public class DrawableGraph
    {
        private readonly List<DrawableNode> _firstLevelNodes = new List<DrawableNode>();

        public void AddFirstLevelNode(DrawableNode node)
        {
            _firstLevelNodes.Add(node);
        }
    }
}