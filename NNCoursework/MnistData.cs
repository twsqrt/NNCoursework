using System.Globalization;
using CsvHelper;
using LinearAlgebra;

namespace NNCoursework;

public static class MnistData
{
    private const string PROJECT_DIRECTORY = @"C:\Users\twsqrt\source\repos\NNCoursework";
    private static string mnistDataDirectory = Path.Combine(PROJECT_DIRECTORY, "MNIST");
    private static string mnistTrainFile = Path.Combine(mnistDataDirectory, "mnist_train.csv");
    private static string mnistTestFile = Path.Combine(mnistDataDirectory, "mnist_test.csv");
    
    const int TRAIN_SIZE = 60000;
    const int TEST_SIZE = 4451;
    const int IMAGE_SIZE = 28 * 28;

    private static Vector[] _trainData;
    private static Vector[] _trainMarkup;
    private static Vector[] _testData;
    private static Vector[] _testMarkup;

    private static void LoadData(int size, string fileName, string dataName, out Vector[] data, out Vector[] markup)
    {
        data = new Vector[size];
        markup = new Vector[size];

        using (var reader = new StreamReader(fileName)) 
        using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture)) 
        { 
            int percentCount = 0;
            for(int i = 0; i < size; i++)
            {
                float percent = 100.0f * i / size;
                if(percent + 1.0f > percentCount)
                {
                    if(percentCount % 5 == 0)
                    {
                        Console.SetCursorPosition(0, Console.CursorTop);
                        Console.Write($"Reading {dataName} data: {percentCount}%");
                    }

                    percentCount++;
                }

                csv.Read();

                int number = csv.GetField<int>(0);
                Vector vector = Vector.CreateZero(10);
                vector[number] = 1.0f;
                markup[i] = vector;

                var dataArray = new float[IMAGE_SIZE];
                for(int j = 0; j < IMAGE_SIZE; j++)
                    dataArray[j] = csv.GetField<float>(j + 1) / 255;

                data[i] = new Vector(dataArray);
            }
        }

        Console.WriteLine();
    }

    public static void LoadTrainData()
        => LoadData(TRAIN_SIZE, mnistTrainFile, "train", out _trainData, out _trainMarkup);
    
    public static void LoadTestData()
        => LoadData(TEST_SIZE, mnistTestFile, "test", out _testData, out _testMarkup);

    public static void GetTrainData(out Vector[] data, out Vector[] markup)
    {
        data = _trainData;
        markup = _trainMarkup;
    }

    public static void GetTestData(out Vector[] data, out Vector[] markup)
    {
        data = _testData;
        markup = _testMarkup;
    }
}
