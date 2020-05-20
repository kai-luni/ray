using Microsoft.VisualStudio.TestTools.UnitTesting;
using ray;
using System;

namespace ray_test
{
    [TestClass]
    public class ForwardPropagationTester
    {
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

            nodeLayerOneOne.addNodeForward(connectorLayerOneTwoNodeOneOne);
            nodeLayerOneOne.addNodeForward(connectorLayerOneTwoNodeOneTwo);
            nodeLayerOneTwo.addNodeForward(connectorLayerOneTwoNodeTwoOne);
            nodeLayerOneTwo.addNodeForward(connectorLayerOneTwoNodeTwoTwo);

            connectorLayerOneTwoNodeOneOne.setNodeForward(nodeLayerTwoOne);
            connectorLayerOneTwoNodeOneTwo.setNodeForward(nodeLayerTwoTwo);
            connectorLayerOneTwoNodeTwoOne.setNodeForward(nodeLayerTwoOne);
            connectorLayerOneTwoNodeTwoTwo.setNodeForward(nodeLayerTwoTwo);

            nodeLayerTwoOne.addNodeBackward(connectorLayerOneTwoNodeOneOne);
            nodeLayerTwoOne.addNodeBackward(connectorLayerOneTwoNodeTwoOne);
            nodeLayerTwoTwo.addNodeBackward(connectorLayerOneTwoNodeOneTwo);
            nodeLayerTwoTwo.addNodeBackward(connectorLayerOneTwoNodeTwoTwo);

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

            nodeLayerOneOne.addNodeForward(connectorLayerOneTwoNodeOneOne);
            nodeLayerOneOne.addNodeForward(connectorLayerOneTwoNodeOneTwo);
            nodeLayerOneTwo.addNodeForward(connectorLayerOneTwoNodeTwoOne);
            nodeLayerOneTwo.addNodeForward(connectorLayerOneTwoNodeTwoTwo);

            connectorLayerOneTwoNodeOneOne.setNodeForward(nodeLayerTwoOne);
            connectorLayerOneTwoNodeOneTwo.setNodeForward(nodeLayerTwoTwo);
            connectorLayerOneTwoNodeTwoOne.setNodeForward(nodeLayerTwoOne);
            connectorLayerOneTwoNodeTwoTwo.setNodeForward(nodeLayerTwoTwo);

            connectorLayerOneTwoNodeOneOne.setNodeBackward(nodeLayerOneOne);
            connectorLayerOneTwoNodeOneTwo.setNodeBackward(nodeLayerOneOne);
            connectorLayerOneTwoNodeTwoOne.setNodeBackward(nodeLayerOneTwo);
            connectorLayerOneTwoNodeTwoTwo.setNodeBackward(nodeLayerOneTwo);

            nodeLayerTwoOne.addNodeBackward(connectorLayerOneTwoNodeOneOne);
            nodeLayerTwoOne.addNodeBackward(connectorLayerOneTwoNodeTwoOne);
            nodeLayerTwoTwo.addNodeBackward(connectorLayerOneTwoNodeOneTwo);
            nodeLayerTwoTwo.addNodeBackward(connectorLayerOneTwoNodeTwoTwo);

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
