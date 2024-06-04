using System.Globalization;
using CsvHelper;
using LinearAlgebra;
using NeuralNetworks;
using NeuralNetworks.Activation;
using NeuralNetworks.ComputationGraph;
using NeuralNetworks.Network;
using NNCoursework;

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
                    {
                        Console.WriteLine($"Reading data progress: {percentCount}%");
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

            Console.WriteLine();
        }
    }

    public static NeuralNetwork TrainNetwork(int numberOfEpochs, float learningRate, float weigthDecay)
    {
        Vector[] data, markup;
        ReadData(out data, out markup);

        var input = new VectorInputNode(IMAGE_SIZE);
        var inputMat = new ReshapeNode<Vector, Matrix>(input, new TensorShape(28, 28));
        var kernel = new DataNode<Tensor3D>(new TensorShape(11, 11, 8));
        var conv = new Convolution2DNode(inputMat, kernel, false);
        var maxPool = new MaxPool2DNode(conv, 2, 2);
        var convOutput = new FlattenNode<Tensor3D>(maxPool);

        var weights1 = new DataNode<Matrix>(new TensorShape(100, convOutput.Shape.Dimension));
        var layer1 = new LayerNode(weights1, convOutput);
        var bias1 = new DataNode<Vector>(new TensorShape(100));
        var add1 = new AdditionNode<Vector>(layer1, bias1);
        var act1 = ActivationNode<Vector>.Create(add1, ActivationType.LOGSIG); 

        var weights2 = new DataNode<Matrix>(new TensorShape(30, 100));
        var layer2 = new LayerNode(weights2, act1);
        var bias2 = new DataNode<Vector>(new TensorShape(30));
        var add2 = new AdditionNode<Vector>(layer2, bias2);
        var act2 = ActivationNode<Vector>.Create(add2, ActivationType.LOGSIG);

        var weights3 = new DataNode<Matrix>(new TensorShape(10, 30));
        var layer3 = new LayerNode(weights3, act2);
        var bias3 = new DataNode<Vector>(new TensorShape(10));
        var add3 = new AdditionNode<Vector>(layer3, bias3);
        var output = ActivationNode<Vector>.Create(add3, ActivationType.LOGSIG);

        var network = new NeuralNetwork(input, 
            new IDataNode[] {kernel, weights1, bias1, weights2, bias2, weights3, bias3}, 
            output);

        for(int i = 0; i < numberOfEpochs; i++)
        {
            Console.WriteLine($"Epoch: {i + 1} / {numberOfEpochs}");

            network.Fit(data, markup, learningRate, weigthDecay, Console.Out);

            Console.WriteLine($"Train rate: {Test.NetworkRate(network, data, markup)}");
            Console.WriteLine($"Test rate: {Test.NetworkRateOnTest(network)}");
        }
        return network;
    }
}
