using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Keras;
using Keras.Layers;
using Keras.Models;
using Numpy;

namespace ray
{
    class Program
    {
        static void Main(string[] args)
        {
            var rand = new Random(); 
            var layer_sizes = new List<int>(){2, 5, 5, 2};
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


            var x = new List<List<double>>(){};
            x.Add(new List<double>(){0.0, 0.0});
            x.Add(new List<double>(){0.0, 1.0});
            x.Add(new List<double>(){1.0, 0.0});
            x.Add(new List<double>(){1.0, 1.0});
            var y = new List<List<double>>(){};
            y.Add(new List<double>(){0.0, 0.0});
            y.Add(new List<double>(){1.0, 1.0});
            y.Add(new List<double>(){1.0, 1.0});
            y.Add(new List<double>(){0.0, 0.0});

            int iterations = 1000000;
            var smallest_error = RayTrainer.Train(ref neural_net, x, y, iterations, debug_output: true);
            Console.WriteLine($"Smallest error: {smallest_error}");



            // var x = np.array(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
            // var y = np.array(new float[] { 0, 1, 1, 0 });

            // //Build sequential model
            // var model = new Sequential();
            // model.Add(new Dense(32, activation: "relu", input_shape: new Shape(2)));
            // model.Add(new Dense(64, activation: "relu"));
            // model.Add(new Dense(1, activation: "sigmoid"));

            // //Compile and train
            // model.Compile(optimizer: "sgd", loss: "binary_crossentropy", metrics: new string[] { "accuracy" });
            // model.Fit(x, y, batch_size: 2, epochs: 1000, verbose: 1);

            // //Save model and weights
            // string json = model.ToJson();
            // File.WriteAllText("model.json", json);
            // model.SaveWeight("model.h5");

            // //Load model and weight
            // var loaded_model = Sequential.ModelFromJson(File.ReadAllText("model.json"));
            // loaded_model.LoadWeight("model.h5");
        }
    }
}
