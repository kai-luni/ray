using System;
using System.Collections.Generic;

namespace ray.helper;

public class ModelHelper
{
    /// <summary>
    /// Xavier weight initialization method for neural networks. This method helps to keep the scale of the gradients roughly the same in all layers.
    /// It is particularly useful for deep networks, where the gradients can vanish or explode during training
    /// </summary>
    /// <param name="random">this is a random number generator</param>
    /// <param name="fanIn">this is the number of input units</param>
    /// <param name="fanOut">this is the number of output units</param>
    /// <returns></returns>
    public static double XavierWeight(Random random, int fanIn, int fanOut)
    {
        double limit = Math.Sqrt(6.0 / (fanIn + fanOut));
        return (random.NextDouble() * 2.0 - 1.0) * limit;
    }

    /// <summary>
    /// Xavier weight initialization for all layers of a neural network. This method generates weights for each layer based on the number of input and output units, ensuring that the weights are initialized in a way that helps maintain the scale of the gradients during training.
    /// </summary>
    /// <param name="random">this is a random number generator</param>
    /// <param name="layerSizes">this is a list of integers representing the size of each layer</param>
    /// <returns></returns>
    public static List<List<double>> XavierWeights(Random random, List<int> layerSizes)
    {
        var weights = new List<List<double>>();
        for (int i = 0; i < layerSizes.Count - 1; i++)
        {
            int fanIn = layerSizes[i];
            int fanOut = layerSizes[i + 1];
            var layerWeights = new List<double>();
            for (int j = 0; j < fanIn * fanOut; j++)
            {
                layerWeights.Add(XavierWeight(random, fanIn, fanOut));
            }
            weights.Add(layerWeights);
        }
        return weights;
    }
}