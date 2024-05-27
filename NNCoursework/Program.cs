using System.Globalization;
using CsvHelper;
using LinearAlgebra;
using NeuralNetworks.Activation;
using NeuralNetworks.SDGMethod;
using NeuralNetworks.Network;

const string PROJECT_DIRECTORY = @"C:\Users\twsqrt\source\repos\NNCoursework";
string mnistDataDirectory = Path.Combine(PROJECT_DIRECTORY, "MNIST");

string mnistTrainFile = Path.Combine(mnistDataDirectory, "mnist_train.csv");
string mnistTestFile = Path.Combine(mnistDataDirectory, "mnist_test.csv");

const int TRAIN_SIZE = 60000;
const int TEST_SIZE = 4451;
const int IMAGE_SIZE = 28 * 28;

var data = new Vector<float>[TRAIN_SIZE];
var markup = new Vector<float>[TRAIN_SIZE];

using (var reader = new StreamReader(mnistTrainFile)) 
using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture)) 
{ 
    int percentCount = 0;
    for(int i = 0; i < TRAIN_SIZE; i++)
    {
        float percent = 100.0f * i / TRAIN_SIZE;
        if(percent + 1.0f > percentCount)
        {
            if(percentCount % 5 == 0)
                Console.WriteLine($"Read train data progress: {percentCount}%");

            percentCount++;
        }

        csv.Read();

        int number = csv.GetField<int>(0);
        Vector<float> vector = Vector<float>.CreateZeroVector(10);
        vector[number] = 1.0f;
        markup[i] = vector;

        var dataArray = new float[IMAGE_SIZE];
        for(int j = 0; j < IMAGE_SIZE; j++)
            dataArray[j] = csv.GetField<float>(j + 1) / 255;

        data[i] = new Vector<float>(dataArray);
    }
}

var builder = new NetworkBuilder();
Network network = builder.Create()
    .WithInput(IMAGE_SIZE)
    .ToLayer(200)
    .WithActivationFunction(ActivationType.LOGSIG)
    .ToLayer(80)
    .WithActivationFunction(ActivationType.LOGSIG)
    .ToLayer(10)
    .WithActivationFunction(ActivationType.LOGSIG)
    .ToOutput()
    .Build();

var sgdMethod = new RegularSGD(0.01f);
network.Fit(data, markup, sgdMethod, 5, Console.Out);

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
        
        var input = new Vector<float>(dataArray);

        Vector<float> output = network.Execute(input);
        if(float.IsNaN(output.LengthSquared))
            throw new ArgumentException();

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

float correctRate = (float) correctCount / TEST_SIZE;
Console.WriteLine($"Correct rate: {correctRate}");