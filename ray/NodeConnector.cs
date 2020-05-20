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
            //TODO: learning rate public static
            double learningRate = 0.01;
            this.weight += nodeBackward.finalValue * weightUpdateValueTwo * learningRate;
            //pass the error on backward
            nodeBackward.Backpropagate(errorValue);
            
        }
    }
}
