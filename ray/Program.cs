using System;
using System.Collections.Generic;
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

        var (Images, Labels) = MnistLoader.Load(
            "data/train-images-idx3-ubyte.gz",
            "data/train-labels-idx1-ubyte.gz",
            maxSamples: 12000
        );

        var testData = MnistLoader.Load(
            "data/t10k-images-idx3-ubyte.gz",
            "data/t10k-labels-idx1-ubyte.gz",
            maxSamples: 1000
        );

        Console.WriteLine(
            $"Training samples: {Images.Count}"
        );

        Console.WriteLine(
            $"Test samples: {testData.Images.Count}"
        );

        //
        // 2. Netzwerk aufbauen
        //
        // MNIST:
        // 28 x 28 Pixel = 784 Inputs
        //
        // Architektur:
        // 784 -> 32 -> 10
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

        double initialAccuracy = Evaluate(
            neuralNet,
            testData.Images,
            testData.Labels
        );

        //
        // 4. Training
        //

        int epochs = 5;

        Console.WriteLine();
        Console.WriteLine("Training beginnt...");
        Console.WriteLine();

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            //
            // Eine Iteration von RayTrainer entspricht hier
            // einer vollständigen Epoche über alle Samples.
            //

            double error = RayTrainer.Train(
                ref neuralNet,
                Images,
                Labels,
                iterations: 1,
                debug_output: false
            );

            double accuracy = CalculateAccuracy(
                neuralNet,
                testData.Images,
                testData.Labels
            );

            Console.WriteLine(
                $"Epoch {epoch}/{epochs} - " +
                $"Error: {error:F6} - " +
                $"Test Accuracy: {accuracy:F2} %"
            );
        }

        //
        // 5. Abschließende Auswertung
        //

        Console.WriteLine();
        Console.WriteLine("Nach Training:");

        double finalAccuracy = Evaluate(
            neuralNet,
            testData.Images,
            testData.Labels
        );

        Console.WriteLine();
        Console.WriteLine(
            $"Verbesserung: " +
            $"{initialAccuracy:F2} % -> {finalAccuracy:F2} %"
        );

        //
        // 6. Einige einzelne Vorhersagen anzeigen
        //

        Console.WriteLine();
        Console.WriteLine("Beispielvorhersagen:");
        Console.WriteLine();

        ShowPredictions(
            neuralNet,
            testData.Images,
            testData.Labels,
            count: 20
        );
    }


    private static double Evaluate(
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

        return accuracy;
    }


    private static double CalculateAccuracy(
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

        return (double)correct / images.Count * 100.0;
    }


    private static void ShowPredictions(
        NeuralNet neuralNet,
        List<List<double>> images,
        List<List<double>> labels,
        int count)
    {
        int samplesToShow =
            Math.Min(count, images.Count);

        for (int i = 0; i < samplesToShow; i++)
        {
            List<double> outputs =
                neuralNet.ForwardValues(images[i]);

            int prediction = ArgMax(outputs);
            int expected = ArgMax(labels[i]);

            string result =
                prediction == expected
                    ? "OK"
                    : "FALSCH";

            Console.WriteLine(
                $"Sample {i,3}: " +
                $"Soll={expected}, " +
                $"Vorhersage={prediction} " +
                $"[{result}]"
            );
        }
    }


    private static int ArgMax(
        List<double> values)
    {
        if (values.Count == 0)
        {
            throw new ArgumentException(
                "ArgMax benötigt mindestens einen Wert.",
                nameof(values)
            );
        }

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