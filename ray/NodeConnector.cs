using System;
using System.Collections.Generic;
using System.Text;

namespace ray
{
    public class NodeConnector
    {
        //debug mode
        public bool debug = false; 

        private readonly double learning_rate;

        //name of the node
        public string name;

        //all the nodes
        PropagationNode nodeForward;
        PropagationNode nodeBackward;

        //experimental
        public double out_value;

        //weight of this connection
        public double weight;
        public double errorBackProp;

        public NodeConnector(double weight, string name = "noname", double learning_rate = 0.1)
        {
            this.weight = weight;
            this.name = name;
            this.learning_rate = learning_rate;
        }

        /// <summary>
        /// connect all_nodes layers with all_nod_connectors (weights)
        /// </summary>
        /// <param name="all_nodes">the nodes</param>
        /// <param name="all_node_connectors">the weights</param>
        public static void AddAllNodeConnectors(ref List<List<PropagationNode>> all_nodes, ref List<List<NodeConnector>> all_node_connectors)
        {
            //add connections between nodes
            for (int i = 0; i < all_node_connectors.Count; i++)
            {   
                var nodes = all_nodes[i];
                var nodeConnectors = all_node_connectors[i];
                var nodesNextLevel = all_nodes[i + 1];

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
            if(debug)
            {
                Console.WriteLine($"NodeConnector {this.name}: Forward weigth {this.weight} * value {value} = {this.weight * value}");
            }
            this.out_value = value;
            this.nodeForward.AddToValue(this.weight * value);
        }

        public void Backpropagate(double error_weight)
        {
            if (debug)
            {
                Console.WriteLine($"NodeConnector {this.name}: Backpropagate: New Weight is {this.weight - this.learning_rate * error_weight}, calculate {this.weight} -= {this.learning_rate} * {error_weight}");
            }

            this.weight -= this.learning_rate * error_weight;

            // this.errorBackProp = errorValue;
            // //update the weight
            // this.weight += nodeBackward.finalValue * weightUpdateValueTwo * LearningParameters.LearningRate;
            //pass the error on backward
            nodeBackward.Backpropagate(error_weight);
            
        }
    }
}
