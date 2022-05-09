using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace ray
{
    public class PropagationNode
    {
        //bias
        private readonly double bias;

        //debug mode
        public bool debug = false;

        //the value of this node, it will be forwarded to the next nodes when all messages arrived
        private List<double> values;
        //the final value is calculated when all values from layer before arrived, its public for debugging purposes
        public double finalValue;
        //counter of messages arrived
        private int messagesArrivedForward;
        private int messagesArrivedFromForward;

        //layer the node is in
        private readonly int layer;

        // name of the node
        private readonly string name;

        public double expectedFinalValue;
        public double errorBackProp;
        //all the backward nodes
        List<NodeConnector> connectorBackward;
        //all the forward nodes
        public List<NodeConnector> connectorForward;

        public PropagationNode(int layer, double bias, string weight_name = "noname")
        {
            this.bias = bias;
            this.name = weight_name;
            this.layer = layer;
            this.connectorBackward = new List<NodeConnector>();
            this.connectorForward = new List<NodeConnector>();
            values = new List<double>();
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
            values.Add(valueForward);
            this.messagesArrivedForward++;
            if (this.messagesArrivedForward < connectorBackward.Count)
            {
                return;
            }
            this.messagesArrivedForward = 0;

            this.finalValue = Sigmoid(values.Sum() + bias);
            if(debug)
            {
                Console.WriteLine($"PropNode {this.name}: net_input {values.Sum() + bias} ,final {this.finalValue}, {string.Join(", ", this.values)}");
            }

            foreach (var nodeForward in this.connectorForward)
            {
                nodeForward.ForwardValue(this.finalValue);
            }
            values = new List<double>();

            return;
        }

        /**
         * act as input cell, no activation function and forward directly
         */
        public void ForwardValue(double valueForward)
        {
            this.finalValue = valueForward;
            foreach (var nodeForward in this.connectorForward)
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

            var value_2 = this.finalValue * (1 - this.finalValue);

            //calculate the error for each weight before
            foreach (var nodeBackward in connectorBackward)
            {
                var value_3 = nodeBackward.out_value;
                var error_total = errorBackProp * value_2 * value_3; 

                if (debug)
                {
                    Console.WriteLine($"Backpropagate PropNode {this.name} weight {nodeBackward.name}: value2 {value_2} and value3 {value_3}");
                    Console.WriteLine($"Backpropagate PropNode {this.name} weight {nodeBackward.name}: error {errorBackProp} val2 {value_2} val3 {value_3}");
                }

                nodeBackward.Backpropagate(error_total);
            }

            return;


            // //we need to know all the weights combined
            // double weightsBeforeCombined = connectorBackward.Sum(x => x.weight);

            // //weight update related
            // double weightUpdatePartTwo = finalValue * (1 - finalValue);
            // double weightUpdateBackward = errorBackProp * weightUpdatePartTwo;

            // //calculate the error for each weight before
            // foreach (var nodeBackward in connectorBackward)
            // {
            //     double shareNodeConnection = nodeBackward.weight / weightsBeforeCombined;
            //     double errorConnection = shareNodeConnection * errorBackProp;
            //     nodeBackward.Backpropagate(errorConnection, weightUpdateBackward);
            // }

        }
    }
}
