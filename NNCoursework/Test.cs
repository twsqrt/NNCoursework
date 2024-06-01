using System.Globalization;
using CsvHelper;
using LinearAlgebra;
using NeuralNetworks.Network;

namespace NNCoursework;

public static class Test
{
    private const string PROJECT_DIRECTORY = @"C:\Users\twsqrt\source\repos\NNCoursework";
    private static string mnistDataDirectory = Path.Combine(PROJECT_DIRECTORY, "MNIST");
    private static string mnistTestFile = Path.Combine(mnistDataDirectory, "mnist_test.csv");
    private static string networkFileDirectory = Path.Combine(PROJECT_DIRECTORY, "Networks");

    const int TEST_SIZE = 4451;
    const int IMAGE_SIZE = 28 * 28;

    public static void TestNetwork(NeuralNetwork network)
    {
        int correctCount = 0;
        using (var reader = new StreamReader(mnistTestFile)) 
        using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture)) 
        { 
            for(int i = 0; i < TEST_SIZE; i++)
            {
                csv.Read();

                int number = csv.GetField<int>(0);

                var dataArray = new float[IMAGE_SIZE];
                for(int j = 0; j < IMAGE_SIZE; j++)
                    dataArray[j] = csv.GetField<float>(j + 1) / 255;
                
                var input = new Vector(dataArray);

                Vector output = network.Execute(input);

                float maxElement = 0.0f;
                int outputNumber = 0;
                for(int j = 0; j < output.Dimension; j++)
                {
                    if(maxElement < output[j])
                    {
                        maxElement = output[j];
                        outputNumber = j;
                    }
                }

                if(outputNumber == number)
                    correctCount++;
            }
        }

        float rate = (float) correctCount / TEST_SIZE;
        Console.WriteLine($"Network rate: {rate}");
    }
}
