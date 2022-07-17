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
        //current errorValue for backprop
        private List<double> errorValues = new List<double>();
        //the final value is calculated when all values from layer before arrived
        public double expectedFinalValue;
        public double errorBackProp;
        public double finalValue;
        //counter of messages arrived
        private int messagesArrivedFromForward;

        //layer the node is in
        private readonly int layer;

        // name of the node
        public readonly string name;

        //all the backward nodes
        List<NodeConnector> connectorBackward;
        //all the forward nodes
        public List<NodeConnector> connectorForward;
        //the value of this node, it will be forwarded to the next nodes when all messages arrived
        private List<double> values;

        public PropagationNode(int layer, double bias, string weight_name = "noname")
        {
            this.bias = bias;
            this.name = weight_name;
            this.layer = layer;
            this.connectorBackward = new List<NodeConnector>();
            this.connectorForward = new List<NodeConnector>();
            values = new List<double>();
            this.messagesArrivedFromForward = 0;
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
         * the incoming value here will be stored with other incoming values, once all nodes in the layer before sent
         * their message, the value will be processed with the activation function
         */
        public void AddToValue(double valueForward)
        {
            values.Add(valueForward);
            this.messagesArrivedFromForward++;
            if (this.messagesArrivedFromForward < connectorBackward.Count)
            {
                return;
            }
            this.messagesArrivedFromForward = 0;

            this.finalValue = Sigmoid(values.Sum() + bias);
            if(debug)
            {
                Console.WriteLine($"PropNode {this.name}: net_input {values.Sum() + bias} ,final {this.finalValue}, Values: [{string.Join(", ", this.values)}], bias: {bias}");
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

        /**
         * get forward weight of a node with a certain name
         */
        public double GetWeightForward(string name)
        {
            foreach (var nodeForward in this.connectorForward)
            {
                if (nodeForward.name == name)
                {
                    return nodeForward.weight;
                }
            }
            return 0;
        }

        private double Sigmoid(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }


        /// <summary>
        /// The error is backwarded in shares depending on the weights
        /// of the connections in between
        /// </summary>
        /// <param name="errorFromAhead">error value calculated for this branch from forward</param>
        /// <param name="weightForwardOrig">original weight of the forward connection</param>
        /// <param name="output_node_ahead">the output node ahead in current branch</param>
        public void Backpropagate(double errorFromAhead, double? weightForwardOrig, double? output_node_ahead)
        {
            double output_to_sum = this.finalValue * (1 - this.finalValue);
            double output_ahead_to_sum = output_node_ahead * (1 - output_node_ahead) ?? 1.0;
            double error_temp = output_ahead_to_sum * errorFromAhead * (weightForwardOrig ?? 1.0) * output_to_sum;
            this.errorValues.Add(error_temp);
            if (debug)
            {
                Console.WriteLine($"Backpropagate PropNode {this.name}: PartError {this.errorValues.Count}: {error_temp} = output_ahead_to_sum:{output_ahead_to_sum} * errorFromAhead:{errorFromAhead} * weightForwardOrig:{weightForwardOrig ?? 1.0} * output_to_sum:{output_to_sum}");
            }
            
            if (this.errorValues.Count < connectorForward.Count)
            {
                return;
            }
            messagesArrivedFromForward = 0;

            var value_2 = this.finalValue * (1 - this.finalValue);

            //calculate the error for each weight before
            foreach (var nodeBackward in connectorBackward)
            {
                // var value_3 = nodeBackward.out_value;
                // var error_total = errorBackProp * value_2 * value_3;
                double errorToApplyBackward = 0; 
                for(int i = 0; i < errorValues.Count; i++)
                {
                    var errorTemp = errorValues[i] * nodeBackward.out_value;
                    errorToApplyBackward += errorTemp;
                    if(debug)
                    {
                        Console.WriteLine($"PropNode {this.name}: error {i+1}: error_temp:{errorTemp} = {errorValues[i]} * nodeBackward.out_value:{nodeBackward.out_value}");
                    }
                }
                //if (debug)
                //{
                    //Console.WriteLine($"Backpropagate PropNode {this.name} weight {nodeBackward.name}: error_total {error_total} >> error {errorBackProp} * val2 {value_2} * val3 {value_3}");
                //}

                nodeBackward.Backpropagate(errorToApplyBackward, this.finalValue, errorFromAhead);
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
