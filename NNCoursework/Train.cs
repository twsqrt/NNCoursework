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

        var input = new VectorInputNode(IMAGE_SIZE);
        var inputMat = new ReshapeNode<Vector, Matrix>(input, new TensorShape(28, 28));

        var kernel = new DataNode<Tensor3D>(new TensorShape(5, 5, 4));
        var conv = new Convolution2DNode(inputMat, kernel, false);
        var maxpool = new MaxPool2DNode(conv, 2, 2);
        var convOutput = new ReshapeNode<Tensor3D, Vector>(maxpool, new TensorShape(576));
        
        var weights1 = new DataNode<Matrix>(new TensorShape(150, 576));
        var layer1 = new LayerNode(weights1, convOutput);
        var activation1 = ActivationNode<Vector>.Create(layer1, ActivationType.LOGSIG);

        var weights2 = new DataNode<Matrix>(new TensorShape(50, 150));
        var layer2 = new LayerNode(weights2, activation1);
        var activation2 = ActivationNode<Vector>.Create(layer2, ActivationType.LOGSIG);

        var weights3 = new DataNode<Matrix>(new TensorShape(10, 50));
        var layer3 = new LayerNode(weights3, activation2);
        var activation3 = ActivationNode<Vector>.Create(layer3, ActivationType.LOGSIG);

        var network = new NeuralNetwork(input, 
            new IDataNode[] {kernel, weights1, weights2, weights3}, 
            activation3);

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
