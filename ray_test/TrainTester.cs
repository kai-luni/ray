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

            var connectorLayerOneTwoNodeOneOne = new NodeConnector(0.15, "w1", learning_rate: 0.5);
            var connectorLayerOneTwoNodeTwoOne = new NodeConnector(0.2, "w2", learning_rate: 0.5);
            var connectorLayerOneTwoNodeOneTwo = new NodeConnector(0.25, "w3", learning_rate: 0.5);
            var connectorLayerOneTwoNodeTwoTwo = new NodeConnector(0.3, "w4", learning_rate: 0.5);

            var nodeLayerTwoOne = new PropagationNode(2, bias_one, "h1");
            var nodeLayerTwoTwo = new PropagationNode(2, bias_one, "h2");

            var connectorLayerTwoThreeNodeOneOne = new NodeConnector(0.4, "w5", learning_rate: 0.5);
            var connectorLayerTwoThreeNodeTwoOne = new NodeConnector(0.45, "w6", learning_rate: 0.5);
            var connectorLayerTwoThreeNodeOneTwo = new NodeConnector(0.5, "w7", learning_rate: 0.5);
            var connectorLayerTwoThreeNodeTwoTwo = new NodeConnector(0.55, "w8", learning_rate: 0.5);

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
                for(var i=0; i<result_one.Count; i++)
                {
                    errors.Add(result_one[i] - targets[i]);
                }

                neural_net.Backpropagate(errors);
            }

            Debug.WriteLine("Done");
        }

        /// https://theneuralblog.com/forward-pass-backpropagation-example/
        /// Error is somewhere in the method Backpropagate, the error_total value is not correct, look at w1
        [TestMethod]
        public void TestNeuralBlog()
        {
            double bias_one = 0.25;
            double bias_two = 0.35;
            var biases = new List<double>(){0.0, bias_one, bias_two};
            double learning_rate = 0.6;
            var targets = new List<double>(){0.05, 0.95};
            List<int> layer_sizes = new List<int>(){2,2,2};
            var weights_one_two = new List<double>(){0.1,0.2,0.3,0.4};
            var weights_two_three = new List<double>(){0.5,0.7,0.6,0.8};
            var all_weights = new List<List<double>>(){weights_one_two, weights_two_three};



            var neural_net = new NeuralNet(layer_sizes, all_weights, biases, new List<string>(){"h1", "h2", "o1", "w5", "w6", "w7", "w1", "w2", "w3", "w4"}, learning_rate);
            var outputs = neural_net.ForwardValues(new List<double>(){0.1, 0.5});
            neural_net.Backpropagate(new List<double>(){outputs[0] - targets[0], outputs[1] - targets[1]});
        }


        [TestMethod]
        public void TwoTwoTwoAndTwoSamples()
        {
            double bias_one = 0.35;
            double bias_two = 0.6;
            double learning_rate = 2.2;
            int iterations = 10;

            //init
            var nodeLayerOneOne = new PropagationNode(1, 0.0, "i1");
            var nodeLayerOneTwo = new PropagationNode(1, 0.0, "i2");

            var connectorLayerOneTwoNodeOneOne = new NodeConnector(0.15, "w1", learning_rate: learning_rate);
            var connectorLayerOneTwoNodeTwoOne = new NodeConnector(0.2, "w2", learning_rate: learning_rate);
            var connectorLayerOneTwoNodeOneTwo = new NodeConnector(0.25, "w3", learning_rate: learning_rate);
            var connectorLayerOneTwoNodeTwoTwo = new NodeConnector(0.3, "w4", learning_rate: learning_rate);

            var nodeLayerTwoOne = new PropagationNode(2, bias_one, "h1");
            var nodeLayerTwoTwo = new PropagationNode(2, bias_one, "h2");

            var connectorLayerTwoThreeNodeOneOne = new NodeConnector(0.4, "w5", learning_rate: learning_rate);
            var connectorLayerTwoThreeNodeTwoOne = new NodeConnector(0.45, "w6", learning_rate: learning_rate);
            var connectorLayerTwoThreeNodeOneTwo = new NodeConnector(0.5, "w7", learning_rate: learning_rate);
            var connectorLayerTwoThreeNodeTwoTwo = new NodeConnector(0.55, "w8", learning_rate: learning_rate);

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

            var values_entry = new List<List<double>>(){};
            values_entry.Add(new List<double>(){0.05, 0.1});
            values_entry.Add(new List<double>(){0.20, 0.45});

            var targets = new List<List<double>>(){};
            targets.Add(new List<double>(){0.01, 0.99});
            targets.Add(new List<double>(){0.3, 0.03});

            double smallest_error = 1000.0;
            for (int h=0; h<iterations; h++)
            {
                for(var hh=0; hh<1; hh++)
                {
                    var errors = new List<double>(){};
                    var result_one = neural_net.ForwardValues(values_entry[hh]);
                    for(var i=0; i<result_one.Count; i++)
                    {
                        errors.Add(result_one[i] - targets[hh][i]);
                    }
                    var smallest_error_temp = Math.Abs(errors.Select(x => Math.Abs(x)).Sum() / errors.Count());
                    if (smallest_error_temp < smallest_error)
                    {
                        smallest_error = smallest_error_temp;
                    }

                    neural_net.Backpropagate(errors);
                }
            }

            Debug.WriteLine(smallest_error);
            Assert.AreEqual(0.0, smallest_error, 0.01);
        }


        /// <summary>
        /// two nodes many iterations, see if error small
        /// </summary>
        [TestMethod]
        public void OneOne()
        {
            //init
            var nodeLayerOneOne = new PropagationNode(1, 0.2);

            var connectorLayerOneTwoNodeOneOne = new NodeConnector(0.9, "w1", learning_rate: 0.5);

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
            var neural_net = new NeuralNet(new List<int>(){1,1}, weights, new List<double>(){0.2, 0.4}, new List<string>(), learning_rate: 0.5);

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
            var neural_net = new NeuralNet(new List<int>(){2,2,2}, weights, new List<double>(){0.0, 0.35, 0.6}, new List<string>(){}, learning_rate: 0.5);

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
            var neural_net = new NeuralNet(new List<int>(){2,2,2}, weights, new List<double>(){0.0, 0.35, 0.6}, new List<string>(){}, learning_rate: 0.5);


            var values_entry = new List<double>(){0.05, 0.1};
            var targets = new List<double>(){0.01, 0.99};
            int iterations = 10000;

            var x = new List<List<double>>(){};
            x.Add(values_entry);
            var y = new List<List<double>>(){};
            y.Add(targets);
            var smallest_error = RayTrainer.Train(ref neural_net, x, y, iterations);

            Debug.WriteLine(smallest_error);
            Assert.AreEqual(0.0, smallest_error, 0.01);
        }

        [TestMethod]
        public void ForwardBackwardSixNodesSpecificTwoSamplesTrainTest()
        {
            List<double> weights_layer_one = new List<double>(){0.15, 0.25, 0.20, 0.3};
            List<double> weights_layer_two = new List<double>(){0.4, 0.5, 0.45, 0.55};
            List<List<double>> weights = new List<List<double>>(){};
            weights.Add(weights_layer_one);
            weights.Add(weights_layer_two);
            var neural_net = new NeuralNet(new List<int>(){2,2,2}, weights, new List<double>(){0.0, 0.35, 0.6}, new List<string>(){}, learning_rate: 0.1);


            var values_entry_one = new List<double>(){0.05, 0.1};
            var values_entry_two = new List<double>(){0.1, 0.2};
            var targets_one = new List<double>(){0.01, 0.99};
            var targets_two = new List<double>(){0.69, 0.42};
            int iterations = 10000;

            var x = new List<List<double>>(){};
            x.Add(values_entry_one);
            x.Add(values_entry_two);
            var y = new List<List<double>>(){};
            y.Add(targets_one);
            y.Add(targets_two);
            var smallest_error = RayTrainer.Train(ref neural_net, x, y, iterations);

            Debug.WriteLine(smallest_error);
            Assert.AreEqual(0.0, smallest_error, 0.01);
        }

        [TestMethod]
        public void ForwardBackwardFourLayersTwoSamplesTwoTenTwoTrainTest()
        {
            var rand = new Random(); 
            var layer_sizes = new List<int>(){2, 10, 10, 2};
            List<double> weights_layer_one = new List<double>(){};
            for (int i=0; i<(layer_sizes[0]*layer_sizes[1]); i++)
            {
                weights_layer_one.Add(rand.NextDouble());
            }
            List<double> weights_layer_two = new List<double>(){};
            for (int i=0; i<(layer_sizes[1]*layer_sizes[2]); i++)
            {
                weights_layer_two.Add(rand.NextDouble());
            }
            List<double> weights_layer_three = new List<double>(){};
            for (int i=0; i<(layer_sizes[2]*layer_sizes[3]); i++)
            {
                weights_layer_three.Add(rand.NextDouble());
            }
            List<List<double>> weights = new List<List<double>>(){};
            weights.Add(weights_layer_one);
            weights.Add(weights_layer_two);
            weights.Add(weights_layer_three);
            var neural_net = new NeuralNet(layer_sizes, weights, new List<double>(){0.0, 0.35, 0.4, 0.6}, new List<string>(){}, 0.01);


            var values_entry_one = new List<double>(){0.05, 0.1};
            var values_entry_two = new List<double>(){0.1, 0.2};
            var targets_one = new List<double>(){0.01, 0.99};
            var targets_two = new List<double>(){0.69, 0.42};

            var x = new List<List<double>>(){};
            x.Add(values_entry_one);
            x.Add(values_entry_two);
            var y = new List<List<double>>(){};
            y.Add(targets_one);
            y.Add(targets_two);

            int iterations = 100000;
            var smallest_error = RayTrainer.Train(ref neural_net, x, y, iterations, debug_output: true);

            Debug.WriteLine(smallest_error);
            Assert.AreEqual(0.0, smallest_error, 0.01);
        }

        [TestMethod]
        public void ForwardBackwardDifferentValues()
        {

            var layer_sizes = new List<int>(){2,32,64,1};
            var rand = new Random(); 
            List<double> weights_layer_one = new List<double>(){};
            for (int i=0; i<(layer_sizes[0]*layer_sizes[1]); i++)
            {
                weights_layer_one.Add(rand.NextDouble());
            }
            List<double> weights_layer_two = new List<double>(){};
            for (int i=0; i<(layer_sizes[1]*layer_sizes[2]); i++)
            {
                weights_layer_two.Add(rand.NextDouble());
            }
            List<double> weights_layer_three = new List<double>(){};
            for (int i=0; i<(layer_sizes[2]*layer_sizes[3]); i++)
            {
                weights_layer_three.Add(rand.NextDouble());
            }
            List<List<double>> weights = new List<List<double>>(){};
            weights.Add(weights_layer_one);
            weights.Add(weights_layer_two);
            weights.Add(weights_layer_three);

            var neural_net = new NeuralNet(layer_sizes, weights, new List<double>(){0.0, 0.35, 0.6, 0.3}, new List<string>(){});

            var x = new List<List<double>>(){};
            x.Add(new List<double>(){0, 0});
            x.Add(new List<double>(){0, 1});
            x.Add(new List<double>(){1, 0});
            x.Add(new List<double>(){1, 1});
            var y = new List<List<double>>(){};
            y.Add(new List<double>(){0});
            y.Add(new List<double>(){1});
            y.Add(new List<double>(){1});
            y.Add(new List<double>(){0});
            // var x = np.array(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
            // var y = np.array(new float[] { 0, 1, 1, 0 });
            var smalles_error = RayTrainer.Train(ref neural_net, x, y, 1000000, debug_output: false);
            Debug.WriteLine($"Smallest error: {smalles_error}");

        }
    }
}