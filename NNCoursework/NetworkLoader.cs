using System.Runtime.CompilerServices;
using LinearAlgebra;
using NeuralNetworks;
using NeuralNetworks.Activation;
using NeuralNetworks.ComputationGraph;
using NeuralNetworks.Network;

namespace NNCoursework;

public static class NetworkLoader
{
    const string PROJECT_DIRECTORY = @"C:\Users\twsqrt\source\repos\NNCoursework";
    static string networksFileDirectory = Path.Combine(PROJECT_DIRECTORY, "Networks");

    public static NeuralNetwork CreateNewNetwork()
    {
        var input = new VectorInputNode(28 * 28);

        var inputMat = new ReshapeNode<Vector, Matrix>(input, new TensorShape(28, 28));
        var kernel = new DataNode<Tensor3D>(new TensorShape(9, 9, 6));
        var conv = new Convolution2DNode(inputMat, kernel, false);
        var maxPool = new MaxPool2DNode(conv, 2, 2);
        var convOutput = new FlattenNode<Tensor3D>(maxPool);

        var weights1 = new DataNode<Matrix>(new TensorShape(100, convOutput.Shape.Dimension));
        var layer1 = new LayerNode(weights1, convOutput);
        var bias1 = new DataNode<Vector>(new TensorShape(100));
        var add1 = new AdditionNode<Vector>(layer1, bias1);
        var act1 = ActivationNode<Vector>.Create(add1, ActivationType.LOGSIG); 

        var weights2 = new DataNode<Matrix>(new TensorShape(10, 100));
        var layer2 = new LayerNode(weights2, act1);
        var bias2 = new DataNode<Vector>(new TensorShape(10));
        var add2 = new AdditionNode<Vector>(layer2, bias2);
        var act2 = ActivationNode<Vector>.Create(add2, ActivationType.LOGSIG);

        /*var weights3 = new DataNode<Matrix>(new TensorShape(10, 30));
        var layer3 = new LayerNode(weights3, act2);
        var bias3 = new DataNode<Vector>(new TensorShape(10));
        var add3 = new AdditionNode<Vector>(layer3, bias3);
        var output = ActivationNode<Vector>.Create(add3, ActivationType.LOGSIG);*/

        return new NeuralNetwork(input, 
            new IDataNode[] {kernel, weights1, bias1, weights2, bias2}, 
            act2);
    }

    public static NeuralNetwork LoadShapshot(string fileName)
    {
        string path = Path.Combine(networksFileDirectory, fileName + ".bin");

        using(var stream = File.OpenRead(path))
        using(var reader = new BinaryReader(stream))
            return NeuralNetwork.Import(reader);
    }

    public static void CreateShapshot(NeuralNetwork netowork, string networkName, int currentEpoch)
    {
        string fileName = networkName + "_epoch=" + currentEpoch;
        string path = Path.Combine(networksFileDirectory, fileName + ".bin");

        using(var stream = File.OpenWrite(path))
        using(var writer = new BinaryWriter(stream))
            netowork.Export(writer);
    }
}
