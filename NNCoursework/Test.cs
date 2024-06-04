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
    private static Vector[] _testData;
    private static Vector[] _testMarkup;

    const int TEST_SIZE = 4451;
    const int IMAGE_SIZE = 28 * 28;

    static Test()
    {
        _testData = new Vector[TEST_SIZE];
        _testMarkup = new Vector[TEST_SIZE];

        using (var reader = new StreamReader(mnistTestFile)) 
        using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture)) 
        { 
            for(int i = 0; i < TEST_SIZE; i++)
            {
                csv.Read();

                int number = csv.GetField<int>(0);
                Vector vector = Vector.CreateZero(10);
                vector[number] = 1.0f;
                _testMarkup[i] = vector;

                var dataArray = new float[IMAGE_SIZE];
                for(int j = 0; j < IMAGE_SIZE; j++)
                    dataArray[j] = csv.GetField<float>(j + 1) / 255;
                
                _testData[i] = new Vector(dataArray);
            }
        }
    }

    public static float NetworkRate(NeuralNetwork network, Vector[] data, Vector[] markup)
    {
        int correctCount = 0;

        for(int i = 0; i < TEST_SIZE; i++)
        {
            Vector output = network.Execute(data[i]);

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

            if(markup[i][outputNumber] == 1.0f)
                correctCount++;
        }

        float rate = (float) correctCount / TEST_SIZE;
        return rate;
    }

    public static float NetworkRateOnTest(NeuralNetwork network)
        => NetworkRate(network, _testData, _testMarkup);
}
