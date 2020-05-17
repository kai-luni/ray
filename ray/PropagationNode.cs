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
        private int messagesArrived;
        private bool finalNode;
        public double expectedFinalValue;
        public double errorBackProp;
        //all the backward nodes
        List<NodeConnector> connectorBackward;
        //all the forward nodes
        List<NodeConnector> connectorForward;

        public PropagationNode()
        {
            this.connectorBackward = new List<NodeConnector>();
            this.connectorForward = new List<NodeConnector>();
            value = 0.0f;
            messagesArrived = 0;
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
            messagesArrived++;
            if (messagesArrived != connectorBackward.Count)
            {
                return;
            }
            finalValue = Sigmoid(value);
            if (!finalNode)
            {
                foreach (var nodeForward in connectorForward)
                {
                    nodeForward.ForwardValue(finalValue);
                }
                value = 0.0;
                messagesArrived = 0;
                return;
            }
            double errorValue = expectedFinalValue - finalValue;

        }

        /**
         * act as input cell, no activation function and forward directly
         */
        public void ForwardValue(double valueForward)
        {
            value = 0.0;
            messagesArrived = 0;

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
         * The error is forwarded with appropriate parts to each connection
         */
        public void Backpropagate(double errorValue)
        {
            //we need to know all the weights combined
            double weightsBeforeCombined = connectorBackward.Sum(x => x.weight);

            foreach (var nodeBackward in connectorBackward)
            {
                double shareNodeConnection = nodeBackward.weight / weightsBeforeCombined;
                double shareErrorValue = shareNodeConnection * errorValue;
                nodeBackward.Backpropagate(shareErrorValue);
            }

        }
    }
}
