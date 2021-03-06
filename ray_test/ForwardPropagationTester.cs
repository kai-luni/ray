using Microsoft.VisualStudio.TestTools.UnitTesting;
using ray;
using System;
using System.Collections.Generic;

namespace ray_test
{
    [TestClass]
    public class ForwardPropagationTester
    {
        /**
         * Add a layer of Node Connectors, that connect all the nodes given
         */
        public static void AddNodeConnectors(ref List<PropagationNode> nodes, ref List<NodeConnector> nodeConnectors, ref List<PropagationNode> nodesNextLevel)
        {
            if(nodeConnectors.Count != nodes.Count * nodesNextLevel.Count)
            {
                throw new Exception($"Expected count of Node Connectors: {nodes.Count * nodesNextLevel.Count}, actual count: {nodeConnectors.Count}");
            }

            int counter = 0;
            foreach(var node in nodes)
            {
                foreach(var nodeNextLevel in nodesNextLevel)
                {
                    node.addNodeForward(nodeConnectors[counter]);
                    nodeConnectors[counter].setNodeBackward(node);
                    nodeConnectors[counter].setNodeForward(nodeNextLevel);
                    nodeNextLevel.addNodeBackward(nodeConnectors[counter]);
                    counter++;
                }
            }
        }

        /**
         * check out if forward propagation in a simple neural network
         * with 2 nodes in 2 layers each works correctly
         */
        [TestMethod]
        public void ForwardPropagationFourNodes()
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

            AddNodeConnectors(ref nodesLayerOne, ref nodeConnectorsLayerOneTwo, ref nodesLayerTwo);

            nodeLayerOneOne.ForwardValue(1.0);
            nodeLayerOneTwo.ForwardValue(0.5);

            Assert.AreEqual(0.74, nodeLayerTwoOne.finalValue, 0.1);
            Assert.AreEqual(0.64, nodeLayerTwoTwo.finalValue, 0.1);

        }

        /**
         * backpropagate and see if the errors are calculated correctly
         */
        [TestMethod]
        public void BackwardPropagationFourNodes()
        {
            //init
            var nodeLayerOneOne = new PropagationNode(1);
            var nodeLayerOneTwo = new PropagationNode(1);
            nodeLayerOneOne.finalValue = 1.0;
            nodeLayerOneTwo.finalValue = 1.0;

            var connectorLayerOneTwoNodeOneOne = new NodeConnector(2.0);
            var connectorLayerOneTwoNodeOneTwo = new NodeConnector(1.0);
            var connectorLayerOneTwoNodeTwoOne = new NodeConnector(3.0);
            var connectorLayerOneTwoNodeTwoTwo = new NodeConnector(4.0);

            var nodeLayerTwoOne = new PropagationNode(2);
            var nodeLayerTwoTwo = new PropagationNode(2);
            nodeLayerTwoOne.finalValue = 1.0;
            nodeLayerTwoTwo.finalValue = 1.0;

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

            AddNodeConnectors(ref nodesLayerOne, ref nodeConnectorsLayerOneTwo, ref nodesLayerTwo);

            //backpropagate
            nodeLayerTwoOne.Backpropagate(1.5);
            nodeLayerTwoTwo.Backpropagate(0.5);

            //the errors backward are shared depending on the weight, the higher the weight, the higher 
            // the share of the error
            Assert.AreEqual(0.7, nodeLayerOneOne.errorBackProp, 0.1);
            Assert.AreEqual(1.3, nodeLayerOneTwo.errorBackProp, 0.1);

            //the updated weight here should be slightly larger than 2.0
            Assert.AreEqual(2.0005, connectorLayerOneTwoNodeOneOne.weight, 0.001);
        }

        /**
         * 3 layers with 2 nodes each, forward and backbard propagation
         * for one run
         */
        [TestMethod]
        public void ForwardBackwardSixNodes()
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

            var connectorLayerTwoThreeNodeOneOne = new NodeConnector(0.2);
            var connectorLayerTwoThreeNodeOneTwo = new NodeConnector(0.1);
            var connectorLayerTwoThreeNodeTwoOne = new NodeConnector(0.3);
            var connectorLayerTwoThreeNodeTwoTwo = new NodeConnector(0.4);

            var nodeLayerThreeOne = new PropagationNode(3);
            var nodeLayerThreeTwo = new PropagationNode(3);

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

            AddNodeConnectors(ref nodesLayerOne, ref nodeConnectorsLayerOneTwo, ref nodesLayerTwo);
            AddNodeConnectors(ref nodesLayerTwo, ref nodeConnectorsLayerTwoThree, ref nodesLayerThree);

            //Forward Propagatiom
            nodeLayerOneOne.ForwardValue(1.0);
            nodeLayerOneTwo.ForwardValue(0.5);

            Assert.AreEqual(0.7408, nodeLayerTwoOne.finalValue, 0.01);
            Assert.AreEqual(0.6457, nodeLayerTwoTwo.finalValue, 0.01);

            Assert.AreEqual(0.585, nodeLayerThreeOne.finalValue, 0.01);
            Assert.AreEqual(0.582, nodeLayerThreeTwo.finalValue, 0.01);

            //Backward Propagation
            nodeLayerThreeOne.Backpropagate(0.3);
            nodeLayerThreeTwo.Backpropagate(0.2);

            //TODO: implement the backpropagation connection
            Assert.AreEqual(0.3, nodeLayerThreeOne.errorBackProp, 0.01);
            Assert.AreEqual(0.2, nodeLayerThreeTwo.errorBackProp, 0.01);

            Assert.AreEqual(0.2 + 0.054, connectorLayerTwoThreeNodeOneOne.weight, 0.001);
            Assert.AreEqual(0.1 + 0.036, connectorLayerTwoThreeNodeOneTwo.weight, 0.001);
            Assert.AreEqual(0.3 + 0.047, connectorLayerTwoThreeNodeTwoOne.weight, 0.001);
            Assert.AreEqual(0.4 + 0.031, connectorLayerTwoThreeNodeTwoTwo.weight, 0.001);

            Assert.AreEqual(0.16, nodeLayerTwoOne.errorBackProp, 0.01);
            Assert.AreEqual(0.34, nodeLayerTwoTwo.errorBackProp, 0.01);

            Assert.AreEqual(0.9 + 0.0307, connectorLayerOneTwoNodeOneOne.weight, 0.001);
            Assert.AreEqual(0.2 + 0.0778, connectorLayerOneTwoNodeOneTwo.weight, 0.001);
            Assert.AreEqual(0.3 + 0.0154, connectorLayerOneTwoNodeTwoOne.weight, 0.001);
            Assert.AreEqual(0.8 + 0.0389, connectorLayerOneTwoNodeTwoTwo.weight, 0.001);
        }
    }
}
