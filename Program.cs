using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Onnx;
using Microsoft.ML.OnnxRuntime;
using System.IO;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;

namespace WineClusteringEvaluation
{
    public class WineData
    {
        [LoadColumn(0)] public float Alcohol { get; set; }
        [LoadColumn(1)] public float Malic_Acid { get; set; }
        [LoadColumn(2)] public float Ash { get; set; }
        [LoadColumn(3)] public float Ash_Alcanity { get; set; }
        [LoadColumn(4)] public float Magnesium { get; set; }
        [LoadColumn(5)] public float Total_Phenols { get; set; }
        [LoadColumn(6)] public float Flavanoids { get; set; }
        [LoadColumn(7)] public float Nonflavanoid_Phenols { get; set; }
        [LoadColumn(8)] public float Proanthocyanins { get; set; }
        [LoadColumn(9)] public float Color_Intensity { get; set; }
        [LoadColumn(10)] public float Hue { get; set; }
        [LoadColumn(11)] public float OD280 { get; set; }
        [LoadColumn(12)] public float Proline { get; set; }
    }

    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")] public uint PredictedClusterId { get; set; }
        public float[] Score { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            string dataPath = @"C:\Users\stere\source\repos\Lab5\wine-clustering.csv";
            string modelPath = "wineClusteringModel.zip";
            string onnxModelPath = "wineClusteringModel.onnx"; // Path for the ONNX model

            // Load the dataset
            IDataView data = mlContext.Data.LoadFromTextFile<WineData>(path: dataPath, hasHeader: true, separatorChar: ',');

            // Define feature columns for clustering
            string[] featureColumns = new[] {
                "Alcohol", "Malic_Acid", "Ash", "Ash_Alcanity", "Magnesium",
                "Total_Phenols", "Flavanoids", "Nonflavanoid_Phenols", "Proanthocyanins",
                "Color_Intensity", "Hue", "OD280", "Proline"
            };

            // Define the pipeline for training
            var pipeline = mlContext.Transforms
                .Concatenate("Features", featureColumns)
                .Append(mlContext.Clustering.Trainers.KMeans(featureColumnName: "Features", numberOfClusters: 3));

            // Train the model
            var model = pipeline.Fit(data);

            // Save the model
            mlContext.Model.Save(model, data.Schema, modelPath);
            Console.WriteLine($"Model saved to: {modelPath}");

            // Export the model to ONNX format
            using (var fileStream = new FileStream(onnxModelPath, FileMode.Create))
            {
                mlContext.Model.ConvertToOnnx(model, data, fileStream);
            }
            Console.WriteLine($"Model saved to ONNX format at {onnxModelPath}");

            // Loading the ONNX model for inference
            var session = new InferenceSession(onnxModelPath);

            // Prepare input data for inference
            var inputData = new[]
            {
                new WineData { Alcohol = 13.4f, Malic_Acid = 2.3f, Ash = 2.5f, Ash_Alcanity = 19.8f, Magnesium = 99.5f, Total_Phenols = 2.5f, Flavanoids = 2.3f, Nonflavanoid_Phenols = 0.2f, Proanthocyanins = 1.3f, Color_Intensity = 3.1f, Hue = 0.6f, OD280 = 2.1f, Proline = 1050.0f },
                new WineData { Alcohol = 14.0f, Malic_Acid = 1.8f, Ash = 2.4f, Ash_Alcanity = 20.5f, Magnesium = 99.0f, Total_Phenols = 2.4f, Flavanoids = 2.1f, Nonflavanoid_Phenols = 0.3f, Proanthocyanins = 1.2f, Color_Intensity = 3.0f, Hue = 0.7f, OD280 = 2.0f, Proline = 1030.0f }
            };

            // Convert input data to IDataView
            var newData = mlContext.Data.LoadFromEnumerable(inputData);

            // Prepare the inputs for the ONNX model (using the correct input name based on the metadata)
            var inputTensor = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("Alcohol", new DenseTensor<float>(new[] { 13.4f }, new[] { 1, 1 })),
                NamedOnnxValue.CreateFromTensor("Malic_Acid", new DenseTensor<float>(new[] { 2.3f }, new[] { 1, 1 })),
                NamedOnnxValue.CreateFromTensor("Ash", new DenseTensor<float>(new[] { 2.5f }, new[] { 1, 1 })),
                NamedOnnxValue.CreateFromTensor("Ash_Alcanity", new DenseTensor<float>(new[] { 19.8f }, new[] { 1, 1 })),
                NamedOnnxValue.CreateFromTensor("Magnesium", new DenseTensor<float>(new[] { 99.5f }, new[] { 1, 1 })),
                NamedOnnxValue.CreateFromTensor("Total_Phenols", new DenseTensor<float>(new[] { 2.5f }, new[] { 1, 1 })),
                NamedOnnxValue.CreateFromTensor("Flavanoids", new DenseTensor<float>(new[] { 2.3f }, new[] { 1, 1 })),
                NamedOnnxValue.CreateFromTensor("Nonflavanoid_Phenols", new DenseTensor<float>(new[] { 0.2f }, new[] { 1, 1 })),
                NamedOnnxValue.CreateFromTensor("Proanthocyanins", new DenseTensor<float>(new[] { 1.3f }, new[] { 1, 1 })),
                NamedOnnxValue.CreateFromTensor("Color_Intensity", new DenseTensor<float>(new[] { 3.1f }, new[] { 1, 1 })),
                NamedOnnxValue.CreateFromTensor("Hue", new DenseTensor<float>(new[] { 0.6f }, new[] { 1, 1 })),
                NamedOnnxValue.CreateFromTensor("OD280", new DenseTensor<float>(new[] { 2.1f }, new[] { 1, 1 })),
                NamedOnnxValue.CreateFromTensor("Proline", new DenseTensor<float>(new[] { 1050.0f }, new[] { 1, 1 }))
            };

            // Run inference and handle null values
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputTensor);

            // Process and display prediction results, with null checks
            foreach (var result in results)
            {
                if (result == null)
                {
                    Console.WriteLine("A result was null.");
                    continue;
                }

                // Check if the result value is null
                if (result.Value == null)
                {
                    Console.WriteLine($"{result.Name} returned a null value.");
                    continue;
                }

                // Check if the result can be cast to a tensor and retrieve the data if possible
                if (result.Value is DenseTensor<float> floatTensor)
                {
                    Console.WriteLine($"{result.Name}: {floatTensor[0]}");
                }
                else if (result.Value is DenseTensor<int> intTensor)
                {
                    Console.WriteLine($"{result.Name}: {intTensor[0]}");
                }
                else if (result.Value is DenseTensor<bool> boolTensor)
                {
                    Console.WriteLine($"{result.Name}: {boolTensor[0]}");
                }
                else
                {
                    Console.WriteLine($"{result.Name} has an unexpected type or structure.");
                }
            }

            // Wait for user input before exiting
            Console.WriteLine("Press Enter to exit...");
            Console.ReadLine();
        }
    }
}
