using System;
using System.Collections.Generic;
using System.Linq;
using ray.helper;
using ray.Network;

namespace ray;

class Program
{
    static void Main(string[] args)
    {
        var random = new Random(42);

        //
        // 1. MNIST laden
        //

        var trainData = MnistLoader.Load(
            "data/train-images-idx3-ubyte.gz",
            "data/train-labels-idx1-ubyte.gz",
            maxSamples: 1000
        );

        var testData = MnistLoader.Load(
            "data/t10k-images-idx3-ubyte.gz",
            "data/t10k-labels-idx1-ubyte.gz",
            maxSamples: 200
        );

        Console.WriteLine(
            $"Training samples: {trainData.Images.Count}");

        Console.WriteLine(
            $"Test samples: {testData.Images.Count}");

        //
        // 2. Netzwerk
        //
        // 28 * 28 = 784 Eingabewerte
        // 32 Hidden Nodes
        // 10 Outputs für die Ziffern 0-9
        //

        var layerSizes = new List<int>
        {
            784,
            32,
            10
        };

        var weights = ModelHelper.XavierWeights(
            random,
            layerSizes
        );

        var biases = new List<double>
        {
            0.0,
            0.0,
            0.0
        };

        double learningRate = 0.1;

        var neuralNet = new NeuralNet(
            layerSizes,
            weights,
            biases,
            [],
            learningRate
        );

        //
        // 3. Vor dem Training testen
        //

        Console.WriteLine();
        Console.WriteLine("Vor Training:");

        Evaluate(
            neuralNet,
            testData.Images,
            testData.Labels
        );

        //
        // 4. Training
        //
        // iterations bedeutet bei RayTrainer praktisch:
        // wie oft der komplette Datensatz durchlaufen wird.
        //

        int epochs = 3;

        Console.WriteLine();
        Console.WriteLine("Training...");

        RayTrainer.Train(
            ref neuralNet,
            trainData.Images,
            trainData.Labels,
            epochs,
            debug_output: true
        );

        //
        // 5. Nach dem Training testen
        //

        Console.WriteLine();
        Console.WriteLine("Nach Training:");

        Evaluate(
            neuralNet,
            testData.Images,
            testData.Labels
        );
    }

    /// <summary>
    /// evaluates the neural network on the given images and labels
    /// </summary>
    /// <param name="neuralNet">neural network to evaluate</param>
    /// <param name="images">images to evaluate</param>
    /// <param name="labels"></param>
    private static void Evaluate(
        NeuralNet neuralNet,
        List<List<double>> images,
        List<List<double>> labels)
    {
        int correct = 0;

        for (int i = 0; i < images.Count; i++)
        {
            List<double> outputs =
                neuralNet.ForwardValues(images[i]);

            int prediction = ArgMax(outputs);
            int expected = ArgMax(labels[i]);

            if (prediction == expected)
            {
                correct++;
            }
        }

        double accuracy =
            (double)correct / images.Count * 100.0;

        Console.WriteLine(
            $"Accuracy: {correct}/{images.Count} " +
            $"({accuracy:F2} %)"
        );
    }

    /// <summary>
    /// finds the index of the highest value in a list of doubles
    /// </summary>
    /// <param name="values"></param>
    /// <returns></returns>
    private static int ArgMax(List<double> values)
    {
        int bestIndex = 0;
        double bestValue = values[0];

        for (int i = 1; i < values.Count; i++)
        {
            if (values[i] > bestValue)
            {
                bestValue = values[i];
                bestIndex = i;
            }
        }

        return bestIndex;
    }
}