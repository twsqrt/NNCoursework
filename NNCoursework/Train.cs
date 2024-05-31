using System.Globalization;
using CsvHelper;
using LinearAlgebra;
using NeuralNetworks.Activation;
using NeuralNetworks.SDGMethod;
using NeuralNetworks.Network;

public static class Train
{
    private const string PROJECT_DIRECTORY = @"C:\Users\twsqrt\source\repos\NNCoursework";
    private static string mnistDataDirectory = Path.Combine(PROJECT_DIRECTORY, "MNIST");
    private static string mnistTrainFile = Path.Combine(mnistDataDirectory, "mnist_train.csv");
    private static string networkFileDirectory = Path.Combine(PROJECT_DIRECTORY, "Networks");

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

    public static void TrainNetwork(int numberOfEpochs, float learningRate)
    {

        Vector[] data, markup;
        ReadData(out data, out markup);

        var builder = new NetworkBuilder();
        NeuralNetwork network = builder.Create()
            .WithInput(IMAGE_SIZE)
            .ToLayer(200)
            .WithActivationFunction(ActivationType.LOGSIG)
            .ToLayer(80)
            .WithActivationFunction(ActivationType.LOGSIG)
            .ToLayer(10)
            .WithActivationFunction(ActivationType.LOGSIG)
            .ToOutput()
            .Build();

        var sgdMethod = new RegularSGD(learningRate);

        network.Fit(data, markup, sgdMethod, numberOfEpochs, Console.Out);

        string networkFileName = $"mnist_network_ep{numberOfEpochs}.bin";
        using(var stream = File.OpenWrite(Path.Combine(networkFileDirectory, networkFileName)))
        using(var writer = new BinaryWriter(stream))
            network.Export(writer);
    }
}
