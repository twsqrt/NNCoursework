using System.Globalization;
using CsvHelper;
using LinearAlgebra;
using NeuralNetworks;
using NeuralNetworks.Activation;
using NeuralNetworks.ComputationGraph;
using NeuralNetworks.Network;

public static class Train
{
    private const string PROJECT_DIRECTORY = @"C:\Users\twsqrt\source\repos\NNCoursework";
    private static string mnistDataDirectory = Path.Combine(PROJECT_DIRECTORY, "MNIST");
    private static string mnistTrainFile = Path.Combine(mnistDataDirectory, "mnist_train.csv");

    const int TRAIN_SIZE = 60000;
    const int IMAGE_SIZE = 28 * 28;

    private static void ReadData(out Vector[] data, out Vector[] markup)
    {
        data = new Vector[TRAIN_SIZE];
        markup = new Vector[TRAIN_SIZE];

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
                Vector vector = Vector.CreateZero(10);
                vector[number] = 1.0f;
                markup[i] = vector;

                var dataArray = new float[IMAGE_SIZE];
                for(int j = 0; j < IMAGE_SIZE; j++)
                    dataArray[j] = csv.GetField<float>(j + 1) / 255;

                data[i] = new Vector(dataArray);
            }
        }
    }

    public static NeuralNetwork TrainNetwork(int numberOfEpochs, float learningRate)
    {
        Vector[] data, markup;
        ReadData(out data, out markup);

        var input = new DataNode(IMAGE_SIZE);
        
        var parameter1 = DataNode.CreateRandom(200 * IMAGE_SIZE);
        var reshape1 = new VectorToMatrixNode(parameter1, 200, IMAGE_SIZE);
        var layer1 = new LayerNode(reshape1, input, false);
        var activation1 = ActivationNode.Create(layer1, ActivationType.LOGSIG);

        var parameter2 = DataNode.CreateRandom(80 * 200);
        var reshape2 = new VectorToMatrixNode(parameter2, 80, 200);
        var layer2 = new LayerNode(reshape2, activation1, true);
        var activation2 = ActivationNode.Create(layer2, ActivationType.LOGSIG);

        var parameter3 = DataNode.CreateRandom(80 * 10);
        var reshape3 = new VectorToMatrixNode(parameter3, 10, 80);
        var layer3 = new LayerNode(reshape3, activation2, true);
        var activation3 = ActivationNode.Create(layer3, ActivationType.LOGSIG);

        var network = new NeuralNetwork(input, new[] {parameter1, parameter2, parameter3}, activation3);
        network.Fit(data, markup, learningRate, numberOfEpochs, Console.Out);
        return network;
    }
}
