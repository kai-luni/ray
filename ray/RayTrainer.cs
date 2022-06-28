

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using ray;

public class RayTrainer
{
    public static double Train(ref NeuralNet neural_net, List<List<double>> x, List<List<double>> y, int iterations, bool debug_output = false){
        double smallest_error = 1000.0;
        List<double> debug_errors = new List<double>();
        for (int g=0; g<iterations; g++)
        {
            var errors_iteration = new List<double>();
            for (int h=0; h<x.Count; h++)
            {
                var inputs = x[h];
                var targets = y[h];
                var errors = new List<double>(){};
                var errors_statistics = new List<double>(){};
                var result_one = neural_net.ForwardValues(inputs);
                for(var i=0; i<result_one.Count; i++)
                {
                    errors.Add(result_one[i] - targets[i]);
                    //store absolute error for statistics
                    errors_statistics.Add(Math.Abs(result_one[i] - targets[i]));
                }
                var average_error_sample = errors_statistics.Sum() / errors_statistics.Count;
                errors_iteration.Add(average_error_sample);

                neural_net.Backpropagate(errors);
            }
            var average_error_iteration = errors_iteration.Sum() / errors_iteration.Count;
            if(debug_output)
            {
                debug_errors.Add(average_error_iteration);
            }
            if(debug_output && g%1000==0)
            {
                Console.WriteLine($"Iteration {g} average error: {debug_errors.Sum()/debug_errors.Count}");
                debug_errors = new List<double>();
            }
            if (Math.Abs(average_error_iteration) < smallest_error)
            {
                smallest_error = average_error_iteration;
            }
        }

        return smallest_error;
    }
}