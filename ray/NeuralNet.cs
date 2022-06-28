

using System;
using System.Collections.Generic;
using System.Linq;

namespace ray
{
    public class NeuralNet
    {
        public List<PropagationNode> entryNodes;
        public List<PropagationNode> exitNodes;

        public NeuralNet(ref List<PropagationNode> entryNodes, ref List<PropagationNode> exitNodes)
        {
            this.entryNodes = entryNodes;
            this.exitNodes = exitNodes;
        }

        /// <summary>
        /// initialize the neural net with few parameters
        /// </summary>
        /// <param name="layersizes">size of each layer</param>
        /// <param name="weight_sizes">size of all weights</param>
        /// <param name="biases">bias for each layer</param>
        /// <param name="debugs_entries">list of strings with node debug output, example: w1, o1 ...</param>
        /// <param name="learning_rate">learning rate for training</param>
        public NeuralNet(List<int> layersizes, List<List<double>> weight_sizes, List<double> biases, List<string> debugs_entries, double learning_rate = 0.1)
        {
            if(layersizes.Count != biases.Count)
            {
                throw new Exception("Layers and biases must have the same number of layers.");
            }

            int weight_counter = 1;
            string node_letter = "i";
            var all_nodes = new List<List<PropagationNode>>();
            var all_weights = new List<List<NodeConnector>>();
            for (int i = 0; i < layersizes.Count; i++)
            {
                if(i != 0 && i != (layersizes.Count -1))
                {
                    node_letter = "h";
                }
                else if (i == (layersizes.Count -1))
                {
                    node_letter = "o";
                }
                
                var nodes = new List<PropagationNode>();
                for (int j = 0; j < layersizes[i]; j++)
                {
                    string name = $"{node_letter}{j+1}";
                    var node = new PropagationNode(i+1, biases[i], name);
                    if (debugs_entries.Any(x => x == name))
                    {
                        node.debug = true;
                    }
                    nodes.Add(node);
                }
                all_nodes.Add(nodes);
                if (i == 0) 
                {
                    this.entryNodes = nodes;
                }
                if (i == (layersizes.Count-1))
                {
                    this.exitNodes = nodes;
                }

                //weights
                if (i > (layersizes.Count - 2))
                {
                    continue;
                }

                if (layersizes[i] * layersizes[i + 1] != weight_sizes[i].Count)
                {
                    throw new Exception($"Expected {layersizes[i] * layersizes[i + 1]} weights, but got {weight_sizes[i].Count}.");
                }

                var weights = new List<NodeConnector>();
                //initilize weights for layyer i and i+1
                for (int j = 0; j < (layersizes[i] * layersizes[i + 1]); j++)
                {
                    string name = $"w{weight_counter}";
                    var connector = new NodeConnector(weight_sizes[i][j], name, learning_rate: learning_rate);
                    if (debugs_entries.Any(x => x == name))
                    {
                        connector.debug = true;
                    }
                    weights.Add(connector);
                    weight_counter++;
                }
                all_weights.Add(weights);
            }

            NodeConnector.AddAllNodeConnectors(ref all_nodes, ref all_weights);
        }

        /// <summary>
        /// Forward all values
        /// </summary>
        /// <param name="values">need to have same length as entry nodes</param>
        public List<double> ForwardValues(List<double> values)
        {
            if (values.Count != entryNodes.Count)
            {
                throw new System.Exception("Entry Values need to be same count as Entry Nodes count.");
            }

            for (int i = 0; i < entryNodes.Count; i++)
            {
                entryNodes[i].ForwardValue(values[i]);
            }

            var return_list = new List<double>(){};
            foreach (var exitNode in this.exitNodes)
            {
                return_list.Add(exitNode.finalValue);
            }

            return return_list;
        }

        public void Backpropagate(List<double> errors)
        {
            if (errors.Count != exitNodes.Count)
            {
                throw new System.Exception("Error Values need to be same count as Exit Nodes count.");
            }

            for (int i = 0; i < exitNodes.Count; i++)
            {
                exitNodes[i].Backpropagate(errors[i]);
            }
        }
    }
}