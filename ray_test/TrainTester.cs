

using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Numpy;
using ray;

namespace ray_test
{

    [TestClass]
    public class TrainTester
    {
        public static void Train(Numpy.NDarray<float> x, Numpy.NDarray<float> y){
            for (int i=0; i<x.shape[0]; i++){
                
            }
        }

        [TestMethod]
        public void TwoTwo()
        {
            //init
            var nodeLayerOneOne = new PropagationNode(1);
            var nodeLayerOneTwo = new PropagationNode(1);

            var connectorLayerOneTwoNodeOneOne = new NodeConnector(0.9);
            var connectorLayerOneTwoNodeOneTwo = new NodeConnector(0.2);
            var connectorLayerOneTwoNodeTwoOne = new NodeConnector(0.3);
            var connectorLayerOneTwoNodeTwoTwo = new NodeConnector(0.8);

            var nodeLayerTwoOne = new PropagationNode(2);
            var nodeLayerTwoTwo = new PropagationNode(2);

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

            NodeConnector.AddNodeConnectors(ref nodesLayerOne, ref nodeConnectorsLayerOneTwo, ref nodesLayerTwo);

            var neural_net = new NeuralNet(ref nodesLayerOne, ref nodesLayerTwo);

            var values_entry = new List<double>(){0.5, 0.8};
            var values_exit = new List<double>(){0.4, 0.5};

            for (int h=0; h<300; h++)
            {
                var errors = new List<double>(){};
                var result_one = neural_net.ForwardValues(values_entry);
                for(var i=0; i<result_one.Count; i++)
                {
                    errors.Add((result_one[i] - values_exit[i]));
                }
                Console.WriteLine($"{h}: {string.Join(", ", errors)}");

                neural_net.Backpropagate(errors);
            }

            Console.WriteLine("Done");
        }

        [TestMethod]
        public void OneOne()
        {
            //init
            var nodeLayerOneOne = new PropagationNode(1);

            var connectorLayerOneTwoNodeOneOne = new NodeConnector(0.9);

            var nodeLayerTwoOne = new PropagationNode(2);

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
            var values_exit = new List<double>(){0.4};

            for (int h=0; h<1000; h++)
            {
                var errors = new List<double>(){};
                var result_one = neural_net.ForwardValues(values_entry);
                for(var i=0; i<result_one.Count; i++)
                {
                    errors.Add((result_one[i] - values_exit[i])*-1);
                    if(errors[0] > 0.0 && errors[0] < 0.0009)
                    {
                        Console.WriteLine($">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {neural_net.entryNodes[0].connectorForward[0].weight}");
                    }
                }
                Console.WriteLine($"{h}: {string.Join(", ", errors)}");

                neural_net.Backpropagate(errors);
            }

            Console.WriteLine("Done");
        }
    }
}