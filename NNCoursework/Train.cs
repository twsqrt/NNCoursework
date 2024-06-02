using System.Diagnostics;
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
        var matrixInput = new VectorToMatrixNode(input, 28, 28);

        var parameter0 = DataNode.CreateRandom(5 * 5 * 8);
        var kernel = new VectorToTensor(parameter0, new TensorShape3D(5, 5, 8));
        var conv = new Convolution2DNode(matrixInput, kernel, false);
        var maxpool = new MaxPool2DNode(conv, 2, 2);
        var convOutput = new TensorToVectorNode(maxpool);
        
        var parameter1 = DataNode.CreateRandom(1152 * 100);
        var reshape1 = new VectorToMatrixNode(parameter1, 100, 1152);
        var layer1 = new LayerNode(reshape1, convOutput);
        var activation1 = ActivationNode.Create(layer1, ActivationType.LOGSIG);

        var parameter2 = DataNode.CreateRandom(100 * 10);
        var reshape2 = new VectorToMatrixNode(parameter2, 10, 100);
        var layer2 = new LayerNode(reshape2, activation1);
        var activation2 = ActivationNode.Create(layer2, ActivationType.LOGSIG);

        var network = new NeuralNetwork(input, new[] {parameter0, parameter1, parameter2}, activation2);

        var stopWatch = new Stopwatch();
        stopWatch.Start();

        network.Fit(data, markup, learningRate, numberOfEpochs, Console.Out);

        stopWatch.Stop();
        TimeSpan ts = stopWatch.Elapsed;
        string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
            ts.Hours, ts.Minutes, ts.Seconds,
            ts.Milliseconds / 10);
        Console.WriteLine("Train Time: " + elapsedTime);

        return network;
    }
}
