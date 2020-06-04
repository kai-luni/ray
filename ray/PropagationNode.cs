using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace ray
{
    public class PropagationNode
    {
        //the value of this node, it will be forwarded to the next nodes when all messages arrived
        private double value;
        //the final value is calculated when all values from layer before arrived, its public for debugging purposes
        public double finalValue;
        //counter of messages arrived
        private int messagesArrivedForward;
        private int messagesArrivedFromForward;

        //layer the node is in
        private readonly int layer;

        public double expectedFinalValue;
        public double errorBackProp;
        //all the backward nodes
        List<NodeConnector> connectorBackward;
        //all the forward nodes
        List<NodeConnector> connectorForward;

        public PropagationNode(int layer)
        {
            this.layer = layer;
            this.connectorBackward = new List<NodeConnector>();
            this.connectorForward = new List<NodeConnector>();
            value = 0.0f;
            messagesArrivedForward = 0;
        }

        public void addNodeBackward(NodeConnector connectorBackward)
        {
            this.connectorBackward.Add(connectorBackward);
        }

        public void addNodeForward(NodeConnector connectorForward)
        {
            this.connectorForward.Add(connectorForward);
        }

        /**
         * the incomin value here will be stored with other incoming values, once all nodes in the layer before sent
         * their message, the value will be processes with the activation function
         */
        public void AddToValue(double valueForward)
        {
            value += valueForward;
            messagesArrivedForward++;
            if (messagesArrivedForward < connectorBackward.Count)
            {
                return;
            }
            messagesArrivedForward = 0;

            finalValue = Sigmoid(value);

            foreach (var nodeForward in connectorForward)
            {
                nodeForward.ForwardValue(finalValue);
            }
            value = 0.0;
            return;

        }

        /**
         * act as input cell, no activation function and forward directly
         */
        public void ForwardValue(double valueForward)
        {
            this.finalValue = valueForward;
            foreach (var nodeForward in connectorForward)
            {
                nodeForward.ForwardValue(valueForward);
            }
        }

        private double Sigmoid(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }

        /**
         * The error is backwarded in shares depending on the weights
         * of the connections in between
         */
        public void Backpropagate(double errorValue)
        {
            //for testing TODO: other approach
            errorBackProp += errorValue;
            messagesArrivedFromForward++;
            if (messagesArrivedFromForward < connectorForward.Count)
            {
                return;
            }
            messagesArrivedFromForward = 0;

            //we need to know all the weights combined
            double weightsBeforeCombined = connectorBackward.Sum(x => x.weight);

            //weight update related
            double weightUpdatePartTwo = finalValue * (1 - finalValue);
            double weightUpdateBackward = errorBackProp * weightUpdatePartTwo;

            //calculate the error for each weight before
            foreach (var nodeBackward in connectorBackward)
            {
                double shareNodeConnection = nodeBackward.weight / weightsBeforeCombined;
                double errorConnection = shareNodeConnection * errorBackProp;
                nodeBackward.Backpropagate(errorConnection, weightUpdateBackward);
            }

        }
    }
}
