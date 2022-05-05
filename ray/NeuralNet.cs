

using System.Collections.Generic;

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

            for (int i = 0; i < entryNodes.Count; i++)
            {
                exitNodes[i].Backpropagate(errors[i]);
            }
        }
    }
}