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

        [TestMethod]
        public void ForwardPropagationFourNodes()
        {
            //init
            var nodeLayerOneOne = new PropagationNode();
            var nodeLayerOneTwo = new PropagationNode();

            var connectorLayerOneTwoNodeOneOne = new NodeConnector(0.9);
            var connectorLayerOneTwoNodeOneTwo = new NodeConnector(0.2);
            var connectorLayerOneTwoNodeTwoOne = new NodeConnector(0.3);
            var connectorLayerOneTwoNodeTwoTwo = new NodeConnector(0.8);

            var nodeLayerTwoOne = new PropagationNode();
            var nodeLayerTwoTwo = new PropagationNode();

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
         * Artificially add error in last nodes backpropagate functions and check the error 
         * second last nodes.
         */
        [TestMethod]
        public void BackwardPropagationFourNodes()
        {
            //init
            var nodeLayerOneOne = new PropagationNode();
            var nodeLayerOneTwo = new PropagationNode();
            nodeLayerOneOne.finalValue = 1.0;
            nodeLayerOneTwo.finalValue = 1.0;

            var connectorLayerOneTwoNodeOneOne = new NodeConnector(2.0);
            var connectorLayerOneTwoNodeOneTwo = new NodeConnector(1.0);
            var connectorLayerOneTwoNodeTwoOne = new NodeConnector(3.0);
            var connectorLayerOneTwoNodeTwoTwo = new NodeConnector(4.0);

            var nodeLayerTwoOne = new PropagationNode();
            var nodeLayerTwoTwo = new PropagationNode();
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

            nodeLayerOneOne.ForwardValue(1.0);
            nodeLayerOneTwo.ForwardValue(0.5);

            //creating an artificial error and backpropagate it
            double errorOne = nodeLayerTwoOne.finalValue + 1.5;
            double errorTwo = nodeLayerTwoTwo.finalValue + 0.5;
            nodeLayerTwoOne.Backpropagate(1.5);
            nodeLayerTwoTwo.Backpropagate(0.5);

            //the errors backward are shared depending on the weight, the higher the weight, the higher 
            // the share of the error
            Assert.AreEqual(0.7, nodeLayerOneOne.errorBackProp, 0.1);
            Assert.AreEqual(1.3, nodeLayerOneTwo.errorBackProp, 0.1);

            //the updated weight here should be slightly larger than 2.0
            Assert.AreEqual(2.0005, connectorLayerOneTwoNodeOneOne.weight, 0.001);
        }
    }
}
