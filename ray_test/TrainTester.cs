using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ray;
using ray.helper;

namespace ray_test;


[TestClass]
public class TrainTester
{
    /// <summary>
    /// six nodes many iterations larning, see if it can learn the pattern
    /// </summary>
    [TestMethod]
    public void TwoTwoTwo()
    {
        //initialize variables
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

    /// <summary>
    /// check if it runs without crashing
    /// </summary>
    [TestMethod]
    public void TestNeuralBlog()
    {
        double bias_one = 0.25;
        double bias_two = 0.35;
        var biases = new List<double>(){0.0, bias_one, bias_two};
        double learning_rate = 0.6;
        var targets = new List<double>(){0.05, 0.95};
        List<int> layer_sizes = [2,2,2];
        var weights_one_two = new List<double>(){0.1,0.2,0.3,0.4};
        var weights_two_three = new List<double>(){0.5,0.7,0.6,0.8};
        var all_weights = new List<List<double>>(){weights_one_two, weights_two_three};



        var neural_net = new NeuralNet(layer_sizes, all_weights, biases, ["h1", "h2", "o1", "w5", "w6", "w7", "w1"], learning_rate);
        var outputs = neural_net.ForwardValues([0.1, 0.5]);
        Assert.AreEqual(0.734928613, outputs[0], 0.000001);
        Assert.AreEqual(0.779553884, outputs[1], 0.000001);
        neural_net.Backpropagate([outputs[0] - targets[0], outputs[1] - targets[1]]);
    }


    /// <summary>
    /// check if a small network can learn two samples with two inputs and two outputs
    /// 2-2-2 network, 2 samples, 10000 iterations,
    /// see if the error is small enough
    /// </summary>
    [TestMethod]
    public void TwoTwoTwoAndTwoSamples()
    {
        double bias_one = 0.35;
        double bias_two = 0.6;
        double learning_rate = 0.5;
        int iterations = 10000;

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

        var values_entry = new List<List<double>>
        {
            new() { 0.05, 0.1 },
            new() { 0.20, 0.45 }
        };

        var targets = new List<List<double>>
        {
            new() { 0.01, 0.99 },
            new() { 0.3, 0.03 }
        };

        double smallest_error = 1000.0;
        for (int h=0; h<iterations; h++)
        {
            for(var hh=0; hh < values_entry.Count; hh++)
            {
                var errors = new List<double>(){};
                var result_one = neural_net.ForwardValues(values_entry[hh]);
                for(var i=0; i<result_one.Count; i++)
                {
                    errors.Add(result_one[i] - targets[hh][i]);
                }
                var smallest_error_temp = Math.Abs(errors.Sum(Math.Abs) / errors.Count);
                if (smallest_error_temp < smallest_error)
                {
                    smallest_error = smallest_error_temp;
                }

                neural_net.Backpropagate(errors);
            }
        }

        Debug.WriteLine(smallest_error);
        //Assert.AreEqual(0.0, smallest_error, 0.01);

        double totalAbsoluteError = 0.0;
        int errorCount = 0;

        for (int sample = 0; sample < values_entry.Count; sample++)
        {
            var outputs = neural_net.ForwardValues(values_entry[sample]);

            Console.WriteLine($"Sample {sample}:");

            for (int output = 0; output < outputs.Count; output++)
            {
                double absoluteError =
                    Math.Abs(outputs[output] - targets[sample][output]);

                totalAbsoluteError += absoluteError;
                errorCount++;

                Console.WriteLine(
                    $"  Output {output}: " +
                    $"Ist={outputs[output]:F6}, " +
                    $"Soll={targets[sample][output]:F6}, " +
                    $"Absoluter Fehler={absoluteError:F6}"
                );
            }
        }

        double meanAbsoluteError = totalAbsoluteError / errorCount;

        Console.WriteLine($"Gesamter absoluter Fehler: {totalAbsoluteError:F6}");
        Console.WriteLine($"Mittlerer absoluter Fehler: {meanAbsoluteError:F6}");
    }

    /// <summary>
    /// check out new neural net class with very small net
    /// </summary>
    [TestMethod]
    public void OneOneSmall()
    {            
        List<double> weights_layer_one = [0.9];
        List<List<double>> weights = [weights_layer_one];
        var neural_net = new NeuralNet([1,1], weights, [0.2, 0.4], [], learning_rate: 0.5);

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
        List<double> weights_layer_one = [0.15, 0.25, 0.20, 0.3];
        List<double> weights_layer_two = [0.4, 0.5, 0.45, 0.55];
        List<List<double>> weights = [weights_layer_one, weights_layer_two];
        var neural_net = new NeuralNet([2,2,2], weights, [0.0, 0.35, 0.6], [], learning_rate: 0.5);

        //Forward Propagatiom
        var result = neural_net.ForwardValues([0.05, 0.1]);

        Assert.AreEqual(0.751, result[0], 0.01);
        Assert.AreEqual(0.773, result[1], 0.01);

        //Backpropagation
        var target_o1 = 0.01;
        var target_o2 = 0.99; 

        // //Backward Propagation
        var e1 = result[0] - target_o1;
        var e2 = result[1] - target_o2;
        neural_net.Backpropagate([e1, e2]);

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
        List<double> weights_layer_one = [0.15, 0.25, 0.20, 0.3];
        List<double> weights_layer_two = [0.4, 0.5, 0.45, 0.55];
        List<List<double>> weights = [weights_layer_one, weights_layer_two];
        var neural_net = new NeuralNet([2,2,2], weights, [0.0, 0.35, 0.6], [], learning_rate: 0.5);


        var values_entry = new List<double>(){0.05, 0.1};
        var targets = new List<double>(){0.01, 0.99};
        int iterations = 10000;

        var x = new List<List<double>>
        {
            values_entry
        };
        var y = new List<List<double>>
        {
            targets
        };
        var smallest_error = RayTrainer.Train(ref neural_net, x, y, iterations);

        Debug.WriteLine(smallest_error);
        Assert.AreEqual(0.0, smallest_error, 0.01);
    }

    [TestMethod]
    public void ForwardBackwardSixNodesSpecificTwoSamplesTrainTest()
    {
        List<double> weights_layer_one = [0.15, 0.25, 0.20, 0.3];
        List<double> weights_layer_two = [0.4, 0.5, 0.45, 0.55];
        List<List<double>> weights = [weights_layer_one, weights_layer_two];
        var neural_net = new NeuralNet([2,2,2], weights, [0.0, 0.35, 0.6], [], learning_rate: 0.1);


        var values_entry_one = new List<double>(){0.05, 0.1};
        var values_entry_two = new List<double>(){0.1, 0.2};
        var targets_one = new List<double>(){0.01, 0.99};
        var targets_two = new List<double>(){0.69, 0.42};
        int iterations = 10000;

        var x = new List<List<double>>
        {
            values_entry_one,
            values_entry_two
        };
        var y = new List<List<double>>
        {
            targets_one,
            targets_two
        };
        var smallest_error = RayTrainer.Train(ref neural_net, x, y, iterations);

        Debug.WriteLine(smallest_error);
        Assert.AreEqual(0.0, smallest_error, 0.01);
    }

    /// <summary>
    /// check a larger network if it minimizes the error with two samples
    /// </summary>
    [TestMethod]
    public void ForwardBackwardFourLayersTwoSamplesTwoTenTwoTrainTest()
    {
        var rand = new Random(); 
        var layer_sizes = new List<int>(){2, 10, 10, 2};
        var learning_rate = 0.5;
        int iterations = 100000;

        List<List<double>> weights = ModelHelper.XavierWeights(rand, layer_sizes);
        var neural_net = new NeuralNet(layer_sizes, weights, [0.0, 0.35, 0.4, 0.6], [], learning_rate);


        var values_entry_one = new List<double>(){0.05, 0.1};
        var values_entry_two = new List<double>(){0.1, 0.2};
        var targets_one = new List<double>(){0.01, 0.99};
        var targets_two = new List<double>(){0.69, 0.42};

        var x = new List<List<double>>
        {
            values_entry_one,
            values_entry_two
        };
        var y = new List<List<double>>
        {
            targets_one,
            targets_two
        };
        
        var smallest_error = RayTrainer.Train(ref neural_net, x, y, iterations, debug_output: true);

        Debug.WriteLine(smallest_error);
        Assert.AreEqual(0.0, smallest_error, 0.01);
    }

    // /// <summary>
    // /// check if very large network can learn a pattern with two inputs and one output
    // /// </summary>
    // [TestMethod]
    // public void ForwardBackwardDifferentValues()
    // {
    //     var layer_sizes = new List<int>(){2,32,64,1};
    //     var learning_rate = 0.5;
    //     var rand = new Random(); 
    //     List<List<double>> weights = ModelHelper.XavierWeights(rand, layer_sizes);

    //     var neural_net = new NeuralNet(layer_sizes, weights, [0.0, 0.35, 0.6, 0.3], [], learning_rate);

    //     var x = new List<List<double>>
    //     {
    //         new() { 0, 0 },
    //         new() { 0, 1 },
    //         new() { 1, 0 },
    //         new() { 1, 1 }
    //     };
    //     var y = new List<List<double>>
    //     {
    //         new() { 0 },
    //         new() { 1 },
    //         new() { 1 },
    //         new() { 0 }
    //     };
    //     // var x = np.array(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
    //     // var y = np.array(new float[] { 0, 1, 1, 0 });
    //     var smalles_error = RayTrainer.Train(ref neural_net, x, y, 200000, debug_output: false);
    //     Debug.WriteLine($"Smallest error: {smalles_error}");

    // }
}