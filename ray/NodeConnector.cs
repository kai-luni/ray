using System;
using System.Collections.Generic;
using System.Text;

namespace ray
{
    public class NodeConnector
    {
        //all the nodes
        PropagationNode nodeForward;
        PropagationNode nodeBackward;
        //weight of this connection
        public double weight;
        public double errorBackProp;

        public NodeConnector(double weight)
        { 
            this.weight = weight;
        }

        /**
         * Add a layer of Node Connectors, that connect all the nodes given
         */
        public static void AddNodeConnectors(ref List<PropagationNode> nodes, ref List<NodeConnector> nodeConnectors, ref List<PropagationNode> nodesNextLevel)
        {
            if(nodeConnectors.Count != nodes.Count * nodesNextLevel.Count)
            {
                throw new Exception($"Expected count of Node Connectors: {nodes.Count * nodesNextLevel.Count}, actual count: {nodeConnectors.Count}");
            }

            int counter = 0;
            foreach(var node in nodes)
            {
                foreach(var nodeNextLevel in nodesNextLevel)
                {
                    node.addNodeForward(nodeConnectors[counter]);
                    nodeConnectors[counter].setNodeBackward(node);
                    nodeConnectors[counter].setNodeForward(nodeNextLevel);
                    nodeNextLevel.addNodeBackward(nodeConnectors[counter]);
                    counter++;
                }
            }
        }

        public void setNodeForward(PropagationNode nodeForward)
        {
            this.nodeForward = nodeForward;
        }

        public void setNodeBackward(PropagationNode nodeBackward)
        {
            this.nodeBackward = nodeBackward;
        }

        /**
         * forward the value, multiply it with the weight of this connection
         */
        public void ForwardValue(double value)
        {
            this.nodeForward.AddToValue(this.weight * value);
        }

        public void Backpropagate(double errorValue, double weightUpdateValueTwo)
        {
            this.errorBackProp = errorValue;
            //update the weight
            this.weight += nodeBackward.finalValue * weightUpdateValueTwo * LearningParameters.LearningRate;
            //pass the error on backward
            nodeBackward.Backpropagate(errorValue);
            
        }
    }
}
