

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ray;

namespace ray_test
{

    [TestClass]
    public class TrainTester
    {
        public static double Train(ref NeuralNet neural_net, List<double> inputs, List<double> targets, int iterations){
            double smallest_error = 1000.0;
            for (int h=0; h<iterations; h++)
            {
                var errors = new List<double>(){};
                var result_one = neural_net.ForwardValues(inputs);
                for(var i=0; i<result_one.Count; i++)
                {
                    errors.Add(result_one[i] - targets[i]);
                }
                var average_error = errors.Sum() / errors.Count;
                if (Math.Abs(average_error) < smallest_error)
                {
                    smallest_error = average_error;
                }

                neural_net.Backpropagate(errors);
            }

            return smallest_error;
        }

        // [TestMethod]
        // public void TwoTwo()
        // {
        //     //init
        //     var nodeLayerOneOne = new PropagationNode(1);
        //     var nodeLayerOneTwo = new PropagationNode(1);

        //     var connectorLayerOneTwoNodeOneOne = new NodeConnector(0.9);
        //     var connectorLayerOneTwoNodeOneTwo = new NodeConnector(0.2);
        //     var connectorLayerOneTwoNodeTwoOne = new NodeConnector(0.3);
        //     var connectorLayerOneTwoNodeTwoTwo = new NodeConnector(0.8);

        //     var nodeLayerTwoOne = new PropagationNode(2);
        //     var nodeLayerTwoTwo = new PropagationNode(2);

        //     var nodesLayerOne = new List<PropagationNode>
        //     {
        //         nodeLayerOneOne,
        //         nodeLayerOneTwo
        //     };

        //     var nodeConnectorsLayerOneTwo = new List<NodeConnector>
        //     {
        //         connectorLayerOneTwoNodeOneOne,
        //         connectorLayerOneTwoNodeOneTwo,
        //         connectorLayerOneTwoNodeTwoOne,
        //         connectorLayerOneTwoNodeTwoTwo
        //     };

        //     var nodesLayerTwo = new List<PropagationNode>
        //     {
        //         nodeLayerTwoOne,
        //         nodeLayerTwoTwo
        //     };

        //     NodeConnector.AddNodeConnectors(ref nodesLayerOne, ref nodeConnectorsLayerOneTwo, ref nodesLayerTwo);

        //     var neural_net = new NeuralNet(ref nodesLayerOne, ref nodesLayerTwo);

        //     var values_entry = new List<double>(){0.5, 0.8};
        //     var targets = new List<double>(){0.4, 0.5};

        //     for (int h=0; h<300; h++)
        //     {
        //         var errors = new List<double>(){};
        //         var result_one = neural_net.ForwardValues(values_entry);
        //         for(var i=0; i<result_one.Count; i++)
        //         {
        //             errors.Add((result_one[i] - targets[i]));
        //         }
        //         Console.WriteLine($"{h}: {string.Join(", ", errors)}");

        //         neural_net.Backpropagate(errors);
        //     }

        //     Console.WriteLine("Done");
        // }


        /// <summary>
        /// six nodes many iterations larning, see if it can learn the pattern
        /// </summary>
        [TestMethod]
        public void TwoTwoTwo()
        {
            double bias_one = 0.35;
            double bias_two = 0.6;

            //init
            var nodeLayerOneOne = new PropagationNode(1, 0.0, "i1");
            var nodeLayerOneTwo = new PropagationNode(1, 0.0, "i2");

            var connectorLayerOneTwoNodeOneOne = new NodeConnector(0.15, "w1");
            var connectorLayerOneTwoNodeTwoOne = new NodeConnector(0.2, "w2");
            var connectorLayerOneTwoNodeOneTwo = new NodeConnector(0.25, "w3");
            var connectorLayerOneTwoNodeTwoTwo = new NodeConnector(0.3, "w4");

            var nodeLayerTwoOne = new PropagationNode(2, bias_one, "h1");
            var nodeLayerTwoTwo = new PropagationNode(2, bias_one, "h2");

            var connectorLayerTwoThreeNodeOneOne = new NodeConnector(0.4, "w5");
            var connectorLayerTwoThreeNodeTwoOne = new NodeConnector(0.45, "w6");
            var connectorLayerTwoThreeNodeOneTwo = new NodeConnector(0.5, "w7");
            var connectorLayerTwoThreeNodeTwoTwo = new NodeConnector(0.55, "w8");

            var nodeLayerThreeOne = new PropagationNode(3, bias_two, "o1");
            var nodeLayerThreeTwo = new PropagationNode(3, bias_two, "o2");

            var nodesLayerOne = new List<PropagationNode>
            {
                nodeLayerOneOne,
                nodeLayerOneTwo
            };

            var nodeConnectorsLayerOneTwo = new List<NodeConnector>
            {
                connectorLayerOneTwoNodeOneOne,
                connectorLayerOneTwoNodeOneTwo,
                connectorLayerOneTwoNodeTwoOne,
                connectorLayerOneTwoNodeTwoTwo
            };

            var nodesLayerTwo = new List<PropagationNode>
            {
                nodeLayerTwoOne,
                nodeLayerTwoTwo
            };

            var nodeConnectorsLayerTwoThree = new List<NodeConnector>
            {
                connectorLayerTwoThreeNodeOneOne,
                connectorLayerTwoThreeNodeOneTwo,
                connectorLayerTwoThreeNodeTwoOne,
                connectorLayerTwoThreeNodeTwoTwo
            };

            var nodesLayerThree = new List<PropagationNode>
            {
                nodeLayerThreeOne,
                nodeLayerThreeTwo
            };

            NodeConnector.AddNodeConnectors(ref nodesLayerOne, ref nodeConnectorsLayerOneTwo, ref nodesLayerTwo);
            NodeConnector.AddNodeConnectors(ref nodesLayerTwo, ref nodeConnectorsLayerTwoThree, ref nodesLayerThree);

            var neural_net = new NeuralNet(ref nodesLayerOne, ref nodesLayerThree);

            var values_entry = new List<double>(){0.05, 0.1};
            var targets = new List<double>(){0.01, 0.99};

            for (int h=0; h<10000; h++)
            {
                var errors = new List<double>(){};
                var result_one = neural_net.ForwardValues(values_entry);
                Debug.WriteLine("Next");
                for(var i=0; i<result_one.Count; i++)
                {
                    errors.Add(result_one[i] - targets[i]);

                    Debug.WriteLine($">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {result_one[i]}");
                }

                neural_net.Backpropagate(errors);
            }

            Debug.WriteLine("Done");
        }

        /// <summary>
        /// two nodes many iterations, see if error small
        /// </summary>
        [TestMethod]
        public void OneOne()
        {
            //init
            var nodeLayerOneOne = new PropagationNode(1, 0.2);

            var connectorLayerOneTwoNodeOneOne = new NodeConnector(0.9, "w1");

            var nodeLayerTwoOne = new PropagationNode(2, 0.4);

            var nodesLayerOne = new List<PropagationNode>
            {
                nodeLayerOneOne
            };

            var nodeConnectorsLayerOneTwo = new List<NodeConnector>
            {
                connectorLayerOneTwoNodeOneOne
            };

            var nodesLayerTwo = new List<PropagationNode>
            {
                nodeLayerTwoOne
            };

            NodeConnector.AddNodeConnectors(ref nodesLayerOne, ref nodeConnectorsLayerOneTwo, ref nodesLayerTwo);

            var neural_net = new NeuralNet(ref nodesLayerOne, ref nodesLayerTwo);

            var values_entry = new List<double>(){0.5};
            var targets = new List<double>(){0.4};

            double smallest_error = 1000.0;
            for (int h=0; h<10000; h++)
            {
                var errors = new List<double>(){};
                var result_one = neural_net.ForwardValues(values_entry);
                for(var i=0; i<result_one.Count; i++)
                {
                    errors.Add(result_one[i] - targets[i]);
                }
                if (Math.Abs(errors[0]) < smallest_error)
                {
                    smallest_error = errors[0];
                }

                neural_net.Backpropagate(errors);
            }

            Debug.WriteLine(smallest_error);
            Assert.AreEqual(0.0, smallest_error, 0.01);
        }

        /// <summary>
        /// check out new neural net class with very small net
        /// </summary>
        [TestMethod]
        public void OneOneSmall()
        {            
            List<double> weights_layer_one = new List<double>(){0.9};
            List<List<double>> weights = new List<List<double>>(){};
            weights.Add(weights_layer_one);
            var neural_net = new NeuralNet(new List<int>(){1,1}, weights, new List<double>(){0.2, 0.4}, new List<string>());

            var values_entry = new List<double>(){0.5};
            var targets = new List<double>(){0.4};

            double smallest_error = 1000.0;
            for (int h=0; h<10000; h++)
            {
                var errors = new List<double>(){};
                var result_one = neural_net.ForwardValues(values_entry);
                for(var i=0; i<result_one.Count; i++)
                {
                    errors.Add(result_one[i] - targets[i]);
                }
                if (Math.Abs(errors[0]) < smallest_error)
                {
                    smallest_error = errors[0];
                }

                neural_net.Backpropagate(errors);
            }

            Debug.WriteLine(smallest_error);
            Assert.AreEqual(0.0, smallest_error, 0.01);
        }

        
        /// <summary>
        /// check new neural net class with example from
        /// https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        /// </summary>
        [TestMethod]
        public void ForwardBackwardSixNodesSpecificTest()
        {
            List<double> weights_layer_one = new List<double>(){0.15, 0.25, 0.20, 0.3};
            List<double> weights_layer_two = new List<double>(){0.4, 0.5, 0.45, 0.55};
            List<List<double>> weights = new List<List<double>>(){};
            weights.Add(weights_layer_one);
            weights.Add(weights_layer_two);
            var neural_net = new NeuralNet(new List<int>(){2,2,2}, weights, new List<double>(){0.0, 0.35, 0.6}, new List<string>(){"w1", "w2", "w3", "w4"});

            //Forward Propagatiom
            var result = neural_net.ForwardValues(new List<double>(){0.05, 0.1});

            Assert.AreEqual(0.751, result[0], 0.01);
            Assert.AreEqual(0.773, result[1], 0.01);

            //Backpropagation
            var target_o1 = 0.01;
            var target_o2 = 0.99; 

            // //Backward Propagation
            var e1 = result[0] - target_o1;
            var e2 = result[1] - target_o2;
            neural_net.Backpropagate(new List<double>(){e1, e2});

            foreach(var entry_node in neural_net.entryNodes)
            {
                if(entry_node.name == "i1")
                {
                    var weight_one = entry_node.GetWeightForward("w1");
                    Assert.AreEqual(0.1498, weight_one, 0.01);
                    var weight_two = entry_node.GetWeightForward("w2");
                    Assert.AreEqual(0.2498, weight_two, 0.01);
                } 
                else if (entry_node.name == "i2")
                {
                    var weight_three = entry_node.GetWeightForward("w3");
                    Assert.AreEqual(0.1996, weight_three, 0.01);
                    var weight_four = entry_node.GetWeightForward("w4");
                    Assert.AreEqual(0.2996, weight_four, 0.01);
                }
            }

            //TODO: check if weights are updated correctly
        }

        
        /// <summary>
        /// check a larger network if it minimizes the error
        /// </summary>
        [TestMethod]
        public void ForwardBackwardSixNodesSpecificSuperTrainTest()
        {
            List<double> weights_layer_one = new List<double>(){0.15, 0.25, 0.20, 0.3};
            List<double> weights_layer_two = new List<double>(){0.4, 0.5, 0.45, 0.55};
            List<List<double>> weights = new List<List<double>>(){};
            weights.Add(weights_layer_one);
            weights.Add(weights_layer_two);
            var neural_net = new NeuralNet(new List<int>(){2,2,2}, weights, new List<double>(){0.0, 0.35, 0.6}, new List<string>(){});


            var values_entry = new List<double>(){0.05, 0.1};
            var targets = new List<double>(){0.01, 0.99};
            int iterations = 10000;

            var smallest_error = Train(ref neural_net, values_entry, targets, iterations);

            Debug.WriteLine(smallest_error);
            Assert.AreEqual(0.0, smallest_error, 0.01);
        }
    }
}