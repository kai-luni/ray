using System;
using System.Collections.Generic;
using System.Linq;

namespace ray
{
    public class PropagationNode
    {
        //bias that is also upgraded during backpropagation
        private double _bias;
        //learning rate for backpropagation
        public double _learningRate;

        //debug mode
        public bool debug = false;
        //current errorValue for backprop
        private readonly List<double> _errorValues = [];
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

        /// <summary>
        /// Initializes a new instance of the PropagationNode class with the specified layer, bias, weight name, and learning rate. The node will be part of a neural network and will handle forward and backward propagation of values and errors.
        /// </summary>
        /// <param name="layer"></param>
        /// <param name="bias"></param>
        /// <param name="weight_name"></param>
        /// <param name="learning_rate"></param>
        public PropagationNode(int layer, double bias, string weight_name = "noname", double learning_rate = 0.1)
        {
            _bias = bias;
            _learningRate = learning_rate;
            name = weight_name;
            this.layer = layer;
            connectorBackward = [];
            connectorForward = [];
            values = [];
            messagesArrivedFromForward = 0;
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
            messagesArrivedFromForward++;
            if (messagesArrivedFromForward < connectorBackward.Count)
            {
                return;
            }
            messagesArrivedFromForward = 0;

            finalValue = Sigmoid(values.Sum() + _bias);
            if(debug)
            {
                Console.WriteLine($"PropNode {name}: net_input {values.Sum() + _bias} ,final {finalValue}, Values: [{string.Join(", ", values)}], bias: {_bias}");
            }

            foreach (var nodeForward in connectorForward)
            {
                nodeForward.ForwardValue(finalValue);
            }
            values = [];

            return;
        }

        /**
         * act as input cell, no activation function and forward directly
         */
        public void ForwardValue(double valueForward)
        {
            finalValue = valueForward;
            foreach (var nodeForward in connectorForward)
            {
                nodeForward.ForwardValue(valueForward);
            }
        }

        /**
         * get forward weight of a node with a certain name
         */
        public double GetWeightForward(string name)
        {
            foreach (var nodeForward in connectorForward)
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
        public void Backpropagate(
            double errorFromAhead,
            double? weightForwardOrig,
            double? output_node_ahead)
        {
            double currentNodeDerivative =
                finalValue * (1.0 - finalValue);

            double nextNodeDerivative =
                output_node_ahead.HasValue
                    ? output_node_ahead.Value
                        * (1.0 - output_node_ahead.Value)
                    : 1.0;

            double errorContribution =
                nextNodeDerivative
                * errorFromAhead
                * (weightForwardOrig ?? 1.0)
                * currentNodeDerivative;

            _errorValues.Add(errorContribution);

            int expectedContributions =
                connectorForward.Count == 0
                    ? 1
                    : connectorForward.Count;

            if (_errorValues.Count < expectedContributions)
            {
                return;
            }

            if (_errorValues.Count > expectedContributions)
            {
                throw new InvalidOperationException(
                    $"Backpropagate PropNode {name}: " +
                    $"expected {expectedContributions} error contributions, " +
                    $"but received {_errorValues.Count}."
                );
            }

            try
            {
                // dLoss / dNetInput dieses Knotens
                double nodeDelta = _errorValues.Sum();

                // Input Nodes haben in deinem Aufbau keinen verwendeten Bias.
                if (connectorBackward.Count > 0)
                {
                    double oldBias = _bias;

                    _bias -= _learningRate * nodeDelta;

                    if (debug)
                    {
                        Console.WriteLine(
                            $"PropNode {name}: Bias update: " +
                            $"{oldBias} -= {_learningRate} * {nodeDelta}; " +
                            $"new bias: {_bias}"
                        );
                    }
                }

                foreach (var nodeBackward in connectorBackward)
                {
                    // Gradient eines Gewichts:
                    // Knoten-Delta × Ausgabe des vorherigen Knotens
                    double weightGradient =
                        nodeDelta * nodeBackward.out_value;

                    nodeBackward.Backpropagate(
                        weightGradient,
                        finalValue,
                        errorFromAhead
                    );
                }
            }
            finally
            {
                _errorValues.Clear();
            }
        }
    }
}
