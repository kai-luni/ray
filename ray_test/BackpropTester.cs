using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ray;

namespace ray_test;

[TestClass]
public class BackpropTest
{
    /// <summary>
    /// This test checks if the gradient calculated by backpropagation matches the numerical gradient for a simple neural network with one hidden layer. It uses a small epsilon value to compute the numerical gradient and compares it to the gradient obtained from backpropagation. The test asserts that the two gradients are approximately equal within a specified tolerance.
    /// </summary>
    [TestMethod]
    public void BackpropagationGradientMatchesNumericalGradient()
    {
        const double epsilon = 1e-5;
        const double learningRate = 0.001;

        const double input = 0.5;
        const double target = 0.8;

        const double inputHiddenWeight = 0.4;
        const double hiddenOutputWeight = 0.7;

        // Numerischen Gradienten für das Gewicht hidden -> output berechnen
        double lossPlus = CalculateLoss(
            input,
            target,
            inputHiddenWeight,
            hiddenOutputWeight + epsilon
        );

        double lossMinus = CalculateLoss(
            input,
            target,
            inputHiddenWeight,
            hiddenOutputWeight - epsilon
        );

        double numericalGradient =
            (lossPlus - lossMinus) / (2.0 * epsilon);

        // Frisches Netz mit den unveränderten Gewichten erzeugen
        var inputNode = new PropagationNode(1, 0.0, "i1");
        var hiddenNode = new PropagationNode(2, 0.0, "h1");
        var outputNode = new PropagationNode(3, 0.0, "o1");

        var inputHiddenConnector =
            new NodeConnector(inputHiddenWeight, "w1", learningRate);

        var hiddenOutputConnector =
            new NodeConnector(hiddenOutputWeight, "w2", learningRate);

        var inputNodes = new List<PropagationNode> { inputNode };
        var hiddenNodes = new List<PropagationNode> { hiddenNode };
        var outputNodes = new List<PropagationNode> { outputNode };

        var inputHiddenConnectors =
            new List<NodeConnector> { inputHiddenConnector };

        var hiddenOutputConnectors =
            new List<NodeConnector> { hiddenOutputConnector };

        NodeConnector.AddNodeConnectors(
            ref inputNodes,
            ref inputHiddenConnectors,
            ref hiddenNodes
        );

        NodeConnector.AddNodeConnectors(
            ref hiddenNodes,
            ref hiddenOutputConnectors,
            ref outputNodes
        );

        var neuralNet = new NeuralNet(ref inputNodes, ref outputNodes);

        double originalWeight = hiddenOutputConnector.weight;

        var outputs = neuralNet.ForwardValues([input]);

        neuralNet.Backpropagate([
            outputs[0] - target
        ]);

        double updatedWeight = hiddenOutputConnector.weight;

        // weightNew = weightOld - learningRate * gradient
        double backpropGradient =
            (originalWeight - updatedWeight) / learningRate;

        Console.WriteLine(
            $"Numerical gradient: {numericalGradient:F10}"
        );

        Console.WriteLine(
            $"Backprop gradient:  {backpropGradient:F10}"
        );

        Assert.AreEqual(
            numericalGradient,
            backpropGradient,
            1e-5,
            "Der Backpropagation-Gradient stimmt nicht mit dem numerischen Gradient überein."
        );
    }

    /// <summary>
    /// This test checks if the gradient calculated by backpropagation matches the numerical gradient for the weight connecting the input layer to the hidden layer in a simple neural network. It uses a small epsilon value to compute the numerical gradient and compares it to the gradient obtained from backpropagation. The test asserts that the two gradients are approximately equal within a specified tolerance.
    /// </summary>
    [TestMethod]
    public void BackpropagationGradientMatchesNumericalGradient_ForInputHiddenWeight()
    {
        const double epsilon = 1e-5;
        const double learningRate = 0.001;

        const double input = 0.5;
        const double target = 0.8;

        const double inputHiddenWeight = 0.4;
        const double hiddenOutputWeight = 0.7;

        double lossPlus = CalculateLoss(
            input,
            target,
            inputHiddenWeight + epsilon,
            hiddenOutputWeight
        );

        double lossMinus = CalculateLoss(
            input,
            target,
            inputHiddenWeight - epsilon,
            hiddenOutputWeight
        );

        double numericalGradient =
            (lossPlus - lossMinus) / (2.0 * epsilon);

        var inputNode = new PropagationNode(1, 0.0, "i1");
        var hiddenNode = new PropagationNode(2, 0.0, "h1");
        var outputNode = new PropagationNode(3, 0.0, "o1");

        var inputHiddenConnector =
            new NodeConnector(inputHiddenWeight, "w1", learningRate);

        var hiddenOutputConnector =
            new NodeConnector(hiddenOutputWeight, "w2", learningRate);

        var inputNodes = new List<PropagationNode> { inputNode };
        var hiddenNodes = new List<PropagationNode> { hiddenNode };
        var outputNodes = new List<PropagationNode> { outputNode };

        var inputHiddenConnectors =
            new List<NodeConnector> { inputHiddenConnector };

        var hiddenOutputConnectors =
            new List<NodeConnector> { hiddenOutputConnector };

        NodeConnector.AddNodeConnectors(
            ref inputNodes,
            ref inputHiddenConnectors,
            ref hiddenNodes
        );

        NodeConnector.AddNodeConnectors(
            ref hiddenNodes,
            ref hiddenOutputConnectors,
            ref outputNodes
        );

        var neuralNet = new NeuralNet(ref inputNodes, ref outputNodes);

        double originalWeight = inputHiddenConnector.weight;

        var outputs = neuralNet.ForwardValues([input]);

        neuralNet.Backpropagate([
            outputs[0] - target
        ]);

        double updatedWeight = inputHiddenConnector.weight;

        double backpropGradient =
            (originalWeight - updatedWeight) / learningRate;

        Console.WriteLine(
            $"Numerical gradient: {numericalGradient:F10}"
        );

        Console.WriteLine(
            $"Backprop gradient:  {backpropGradient:F10}"
        );

        Assert.AreEqual(
            numericalGradient,
            backpropGradient,
            1e-5,
            "Der Gradient für Input → Hidden stimmt nicht."
        );
    }

    /// <summary>
    /// This test checks if the gradient calculated by backpropagation matches the numerical gradient for the weight connecting the input layer to the hidden layer in a simple neural network with two output nodes. It uses a small epsilon value to compute the numerical gradient and compares it to the gradient obtained from backpropagation. The test asserts that the two gradients are approximately equal within a specified tolerance.
    /// </summary>
    [TestMethod]
    public void BackpropagationGradientMatchesNumericalGradient_WithTwoOutputs()
    {
        const double epsilon = 1e-5;
        const double learningRate = 0.001;

        const double input = 0.5;

        const double targetOne = 0.8;
        const double targetTwo = 0.2;

        const double inputHiddenWeight = 0.4;
        const double hiddenOutputOneWeight = 0.7;
        const double hiddenOutputTwoWeight = -0.3;

        // Numerischen Gradienten für das Gewicht Input -> Hidden berechnen.
        // Dabei wird der gemeinsame Loss beider Outputs betrachtet.
        double lossPlus = CalculateLossWithTwoOutputs(
            input,
            targetOne,
            targetTwo,
            inputHiddenWeight + epsilon,
            hiddenOutputOneWeight,
            hiddenOutputTwoWeight
        );

        double lossMinus = CalculateLossWithTwoOutputs(
            input,
            targetOne,
            targetTwo,
            inputHiddenWeight - epsilon,
            hiddenOutputOneWeight,
            hiddenOutputTwoWeight
        );

        double numericalGradient =
            (lossPlus - lossMinus) / (2.0 * epsilon);

        // Frisches Netz für den Backpropagation-Gradienten erzeugen.
        var inputNode = new PropagationNode(1, 0.0, "i1");
        var hiddenNode = new PropagationNode(2, 0.0, "h1");

        var outputNodeOne = new PropagationNode(3, 0.0, "o1");
        var outputNodeTwo = new PropagationNode(3, 0.0, "o2");

        var inputHiddenConnector =
            new NodeConnector(
                inputHiddenWeight,
                "w1",
                learning_rate: learningRate
            );

        var hiddenOutputOneConnector =
            new NodeConnector(
                hiddenOutputOneWeight,
                "w2",
                learning_rate: learningRate
            );

        var hiddenOutputTwoConnector =
            new NodeConnector(
                hiddenOutputTwoWeight,
                "w3",
                learning_rate: learningRate
            );

        var inputNodes = new List<PropagationNode>
        {
            inputNode
        };

        var hiddenNodes = new List<PropagationNode>
        {
            hiddenNode
        };

        var outputNodes = new List<PropagationNode>
        {
            outputNodeOne,
            outputNodeTwo
        };

        var inputHiddenConnectors = new List<NodeConnector>
        {
            inputHiddenConnector
        };

        var hiddenOutputConnectors = new List<NodeConnector>
        {
            hiddenOutputOneConnector,
            hiddenOutputTwoConnector
        };

        NodeConnector.AddNodeConnectors(
            ref inputNodes,
            ref inputHiddenConnectors,
            ref hiddenNodes
        );

        NodeConnector.AddNodeConnectors(
            ref hiddenNodes,
            ref hiddenOutputConnectors,
            ref outputNodes
        );

        var neuralNet = new NeuralNet(
            ref inputNodes,
            ref outputNodes
        );

        double originalWeight = inputHiddenConnector.weight;

        var outputs = neuralNet.ForwardValues([input]);

        neuralNet.Backpropagate(
        [
            outputs[0] - targetOne,
            outputs[1] - targetTwo
        ]);

        double updatedWeight = inputHiddenConnector.weight;

        // weightNew = weightOld - learningRate * gradient
        double backpropGradient =
            (originalWeight - updatedWeight) / learningRate;

        Console.WriteLine($"Output 1: {outputs[0]:F10}");
        Console.WriteLine($"Output 2: {outputs[1]:F10}");
        Console.WriteLine($"Numerical gradient: {numericalGradient:F10}");
        Console.WriteLine($"Backprop gradient:  {backpropGradient:F10}");
        Console.WriteLine(
            $"Difference:         " +
            $"{Math.Abs(numericalGradient - backpropGradient):E10}"
        );

        Assert.AreEqual(
            numericalGradient,
            backpropGradient,
            1e-5,
            "Der Input-Hidden-Gradient enthält die Beiträge der beiden " +
            "Output Nodes nicht korrekt."
        );
    }

    /// <summary>
    /// This test checks if the gradients calculated by backpropagation match the numerical gradients for a simple neural network with two input nodes, two hidden nodes, and two output nodes. It uses a small epsilon value to compute the numerical gradients for each weight in the network and compares them to the gradients obtained from backpropagation. The test asserts that all gradients are approximately equal within a specified tolerance.
    /// </summary>
    [TestMethod]
    public void BackpropagationGradientsMatchNumericalGradients_TwoTwoTwoNetwork()
    {
        const double epsilon = 1e-5;
        const double learningRate = 0.001;
        const double tolerance = 1e-5;

        double[] inputs =
        [
            0.3,
            0.7
        ];

        double[] targets =
        [
            0.8,
            0.2
        ];

        // Verbindungsreihenfolge:
        //
        // w1: i1 -> h1
        // w2: i1 -> h2
        // w3: i2 -> h1
        // w4: i2 -> h2
        //
        // w5: h1 -> o1
        // w6: h1 -> o2
        // w7: h2 -> o1
        // w8: h2 -> o2
        double[] initialWeights =
        [
            0.15,
            -0.20,
            0.25,
            0.30,

            0.40,
            -0.45,
            0.50,
            0.55
        ];

        var numericalGradients = new double[initialWeights.Length];

        // Für jedes Gewicht unabhängig den numerischen Gradient berechnen.
        for (int weightIndex = 0;
            weightIndex < initialWeights.Length;
            weightIndex++)
        {
            var weightsPlus =
                (double[])initialWeights.Clone();

            var weightsMinus =
                (double[])initialWeights.Clone();

            weightsPlus[weightIndex] += epsilon;
            weightsMinus[weightIndex] -= epsilon;

            double lossPlus = CalculateTwoTwoTwoLoss(
                inputs,
                targets,
                weightsPlus
            );

            double lossMinus = CalculateTwoTwoTwoLoss(
                inputs,
                targets,
                weightsMinus
            );

            numericalGradients[weightIndex] =
                (lossPlus - lossMinus) / (2.0 * epsilon);
        }

        // Frisches Netz für die echten Backpropagation-Gradienten.
        var setup = BuildTwoTwoTwoNetwork(
            initialWeights,
            learningRate
        );

        NeuralNet neuralNet = setup.Network;
        List<NodeConnector> connectors = setup.Connectors;

        var originalWeights = new double[connectors.Count];

        for (int i = 0; i < connectors.Count; i++)
        {
            originalWeights[i] = connectors[i].weight;
        }

        var outputs = neuralNet.ForwardValues(
            new List<double>
            {
                inputs[0],
                inputs[1]
            }
        );

        neuralNet.Backpropagate(
            new List<double>
            {
                outputs[0] - targets[0],
                outputs[1] - targets[1]
            }
        );

        Console.WriteLine($"Output 1: {outputs[0]:F10}");
        Console.WriteLine($"Output 2: {outputs[1]:F10}");
        Console.WriteLine();

        for (int i = 0; i < connectors.Count; i++)
        {
            double updatedWeight = connectors[i].weight;

            // newWeight = oldWeight - learningRate * gradient
            double backpropGradient =
                (originalWeights[i] - updatedWeight)
                / learningRate;

            double difference = Math.Abs(
                numericalGradients[i] - backpropGradient
            );

            Console.WriteLine(
                $"{connectors[i].name}: " +
                $"numerisch={numericalGradients[i]:F10}, " +
                $"Backprop={backpropGradient:F10}, " +
                $"Differenz={difference:E10}"
            );

            Assert.AreEqual(
                numericalGradients[i],
                backpropGradient,
                tolerance,
                $"Der Gradient von {connectors[i].name} stimmt nicht."
            );
        }
    }

    /// <summary>
    /// This test checks if the weight updates calculated by backpropagation are independent of the order in which the output nodes are processed. It constructs a simple neural network with two input nodes, two hidden nodes, and two output nodes, and performs backpropagation twice: once in normal order (output node 1 first, then output node 2) and once in reversed order (output node 2 first, then output node 1). The test asserts that the final weights after both backpropagation runs are approximately equal within a specified tolerance.
    /// </summary>
    [TestMethod]
    public void BackpropagationDoesNotDependOnOutputProcessingOrder()
    {
        const double learningRate = 0.5;
        const double tolerance = 1e-12;

        double[] inputs =
        [
            0.3,
            0.7
        ];

        double[] targets =
        [
            0.8,
            0.2
        ];

        double[] initialWeights =
        [
            0.15,  // w1: i1 -> h1
            -0.20,  // w2: i1 -> h2
            0.25,  // w3: i2 -> h1
            0.30,  // w4: i2 -> h2

            0.40,  // w5: h1 -> o1
            -0.45,  // w6: h1 -> o2
            0.50,  // w7: h2 -> o1
            0.55   // w8: h2 -> o2
        ];

        double[] weightsAfterNormalOrder =
            ExecuteBackpropagationInOutputOrder(
                initialWeights,
                inputs,
                targets,
                learningRate,
                reverseOutputOrder: false
            );

        double[] weightsAfterReversedOrder =
            ExecuteBackpropagationInOutputOrder(
                initialWeights,
                inputs,
                targets,
                learningRate,
                reverseOutputOrder: true
            );

        for (int i = 0; i < initialWeights.Length; i++)
        {
            double difference = Math.Abs(
                weightsAfterNormalOrder[i]
                - weightsAfterReversedOrder[i]
            );

            Console.WriteLine(
                $"w{i + 1}: " +
                $"normal={weightsAfterNormalOrder[i]:F15}, " +
                $"reversed={weightsAfterReversedOrder[i]:F15}, " +
                $"difference={difference:E10}"
            );

            Assert.AreEqual(
                weightsAfterNormalOrder[i],
                weightsAfterReversedOrder[i],
                tolerance,
                $"Das Ergebnis für w{i + 1} hängt von der " +
                "Verarbeitungsreihenfolge der Output Nodes ab."
            );
        }
    }

    /// <summary>
    /// Calculates the loss for a simple neural network with one hidden layer. The loss is computed as the mean squared error between the output of the network and the target value. The method constructs a small neural network with one input node, one hidden node, and one output node, and performs a forward pass to compute the output. The loss is then calculated based on the difference between the output and the target.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="target"></param>
    /// <param name="inputHiddenWeight"></param>
    /// <param name="hiddenOutputWeight"></param>
    /// <returns></returns>
    private static double CalculateLoss(
        double input,
        double target,
        double inputHiddenWeight,
        double hiddenOutputWeight)
    {
        var inputNode = new PropagationNode(1, 0.0, "i1");
        var hiddenNode = new PropagationNode(2, 0.0, "h1");
        var outputNode = new PropagationNode(3, 0.0, "o1");

        var inputHiddenConnector =
            new NodeConnector(inputHiddenWeight, "w1");

        var hiddenOutputConnector =
            new NodeConnector(hiddenOutputWeight, "w2");

        var inputNodes = new List<PropagationNode> { inputNode };
        var hiddenNodes = new List<PropagationNode> { hiddenNode };
        var outputNodes = new List<PropagationNode> { outputNode };

        var inputHiddenConnectors =
            new List<NodeConnector> { inputHiddenConnector };

        var hiddenOutputConnectors =
            new List<NodeConnector> { hiddenOutputConnector };

        NodeConnector.AddNodeConnectors(
            ref inputNodes,
            ref inputHiddenConnectors,
            ref hiddenNodes
        );

        NodeConnector.AddNodeConnectors(
            ref hiddenNodes,
            ref hiddenOutputConnectors,
            ref outputNodes
        );

        var neuralNet = new NeuralNet(ref inputNodes, ref outputNodes);

        double output = neuralNet.ForwardValues([input])[0];
        double error = output - target;

        // 1/2 * Fehler², damit die Ableitung output - target lautet
        return 0.5 * error * error;
    }

    /// <summary>
    /// Calculates the loss for a simple neural network with one hidden layer and two output nodes. The loss is computed as the sum of the mean squared errors between the outputs of the network and their respective target values. The method constructs a small neural network with one input node, one hidden node, and two output nodes, and performs a forward pass to compute the outputs. The loss is then calculated based on the differences between the outputs and their corresponding targets.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="targetOne"></param>
    /// <param name="targetTwo"></param>
    /// <param name="inputHiddenWeight"></param>
    /// <param name="hiddenOutputOneWeight"></param>
    /// <param name="hiddenOutputTwoWeight"></param>
    /// <returns></returns>
    private static double CalculateLossWithTwoOutputs(
        double input,
        double targetOne,
        double targetTwo,
        double inputHiddenWeight,
        double hiddenOutputOneWeight,
        double hiddenOutputTwoWeight)
    {
        var inputNode = new PropagationNode(1, 0.0, "i1");
        var hiddenNode = new PropagationNode(2, 0.0, "h1");

        var outputNodeOne = new PropagationNode(3, 0.0, "o1");
        var outputNodeTwo = new PropagationNode(3, 0.0, "o2");

        var inputHiddenConnector =
            new NodeConnector(inputHiddenWeight, "w1");

        var hiddenOutputOneConnector =
            new NodeConnector(hiddenOutputOneWeight, "w2");

        var hiddenOutputTwoConnector =
            new NodeConnector(hiddenOutputTwoWeight, "w3");

        var inputNodes = new List<PropagationNode>
        {
            inputNode
        };

        var hiddenNodes = new List<PropagationNode>
        {
            hiddenNode
        };

        var outputNodes = new List<PropagationNode>
        {
            outputNodeOne,
            outputNodeTwo
        };

        var inputHiddenConnectors = new List<NodeConnector>
        {
            inputHiddenConnector
        };

        var hiddenOutputConnectors = new List<NodeConnector>
        {
            hiddenOutputOneConnector,
            hiddenOutputTwoConnector
        };

        NodeConnector.AddNodeConnectors(
            ref inputNodes,
            ref inputHiddenConnectors,
            ref hiddenNodes
        );

        NodeConnector.AddNodeConnectors(
            ref hiddenNodes,
            ref hiddenOutputConnectors,
            ref outputNodes
        );

        var neuralNet = new NeuralNet(
            ref inputNodes,
            ref outputNodes
        );

        var outputs = neuralNet.ForwardValues([input]);

        double errorOne = outputs[0] - targetOne;
        double errorTwo = outputs[1] - targetTwo;

        // Gemeinsamer Loss beider Ausgabeknoten.
        return
            0.5 * errorOne * errorOne +
            0.5 * errorTwo * errorTwo;
    }

    private static double CalculateTwoTwoTwoLoss(
        double[] inputs,
        double[] targets,
        double[] weights)
    {
        var setup = BuildTwoTwoTwoNetwork(
            weights,
            learningRate: 0.0
        );

        var outputs = setup.Network.ForwardValues(
            new List<double>
            {
                inputs[0],
                inputs[1]
            }
        );

        double errorOne = outputs[0] - targets[0];
        double errorTwo = outputs[1] - targets[1];

        return
            0.5 * errorOne * errorOne +
            0.5 * errorTwo * errorTwo;
    }

    private static (
        NeuralNet Network,
        List<NodeConnector> Connectors
    ) BuildTwoTwoTwoNetwork(
        double[] weights,
        double learningRate)
    {
        if (weights.Length != 8)
        {
            throw new ArgumentException(
                $"Es werden genau 8 Gewichte erwartet, " +
                $"erhalten wurden {weights.Length}.",
                nameof(weights)
            );
        }

        const double hiddenBias = 0.1;
        const double outputBias = -0.05;

        var inputNodeOne =
            new PropagationNode(1, 0.0, "i1");

        var inputNodeTwo =
            new PropagationNode(1, 0.0, "i2");

        var hiddenNodeOne =
            new PropagationNode(2, hiddenBias, "h1");

        var hiddenNodeTwo =
            new PropagationNode(2, hiddenBias, "h2");

        var outputNodeOne =
            new PropagationNode(3, outputBias, "o1");

        var outputNodeTwo =
            new PropagationNode(3, outputBias, "o2");

        var connectorOne = new NodeConnector(
            weights[0],
            "w1",
            learningRate
        );

        var connectorTwo = new NodeConnector(
            weights[1],
            "w2",
            learningRate
        );

        var connectorThree = new NodeConnector(
            weights[2],
            "w3",
            learningRate
        );

        var connectorFour = new NodeConnector(
            weights[3],
            "w4",
            learningRate
        );

        var connectorFive = new NodeConnector(
            weights[4],
            "w5",
            learningRate
        );

        var connectorSix = new NodeConnector(
            weights[5],
            "w6",
            learningRate
        );

        var connectorSeven = new NodeConnector(
            weights[6],
            "w7",
            learningRate
        );

        var connectorEight = new NodeConnector(
            weights[7],
            "w8",
            learningRate
        );

        var inputNodes = new List<PropagationNode>
        {
            inputNodeOne,
            inputNodeTwo
        };

        var hiddenNodes = new List<PropagationNode>
        {
            hiddenNodeOne,
            hiddenNodeTwo
        };

        var outputNodes = new List<PropagationNode>
        {
            outputNodeOne,
            outputNodeTwo
        };

        var inputHiddenConnectors = new List<NodeConnector>
        {
            connectorOne,
            connectorTwo,
            connectorThree,
            connectorFour
        };

        var hiddenOutputConnectors = new List<NodeConnector>
        {
            connectorFive,
            connectorSix,
            connectorSeven,
            connectorEight
        };

        NodeConnector.AddNodeConnectors(
            ref inputNodes,
            ref inputHiddenConnectors,
            ref hiddenNodes
        );

        NodeConnector.AddNodeConnectors(
            ref hiddenNodes,
            ref hiddenOutputConnectors,
            ref outputNodes
        );

        var network = new NeuralNet(
            ref inputNodes,
            ref outputNodes
        );

        var allConnectors = new List<NodeConnector>
        {
            connectorOne,
            connectorTwo,
            connectorThree,
            connectorFour,
            connectorFive,
            connectorSix,
            connectorSeven,
            connectorEight
        };

        return (network, allConnectors);
    }

    private static double[] ExecuteBackpropagationInOutputOrder(
        double[] initialWeights,
        double[] inputs,
        double[] targets,
        double learningRate,
        bool reverseOutputOrder)
    {
        var setup = BuildTwoTwoTwoNetwork(
            initialWeights,
            learningRate
        );

        NeuralNet neuralNet = setup.Network;
        List<NodeConnector> connectors = setup.Connectors;

        var outputs = neuralNet.ForwardValues(
        [
            inputs[0],
            inputs[1]
        ]);

        double[] errors =
        [
            outputs[0] - targets[0],
            outputs[1] - targets[1]
        ];

        if (reverseOutputOrder)
        {
            // Zuerst o2, danach o1
            neuralNet.exitNodes[1].Backpropagate(
                errors[1],
                null,
                null
            );

            neuralNet.exitNodes[0].Backpropagate(
                errors[0],
                null,
                null
            );
        }
        else
        {
            // Zuerst o1, danach o2
            neuralNet.exitNodes[0].Backpropagate(
                errors[0],
                null,
                null
            );

            neuralNet.exitNodes[1].Backpropagate(
                errors[1],
                null,
                null
            );
        }

        return connectors
            .Select(connector => connector.weight)
            .ToArray();
    }
}

