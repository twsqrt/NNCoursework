using LinearAlgebra;
using NeuralNetworks.Network;
using NeuralNetworks.Transfer;

var builder = new NetworkBuilder();
Network network = builder.Create()
    .WithInput(5)
    .ToLayerWithoutBias(10)
    .WithTransferFunction(ActivationFunctionType.SATLINS)
    .ToLayerWithoutBias(10)
    .WithTransferFunction(ActivationFunctionType.LOGSIG)
    .ToOutput()
    .Build();

//var inputValue = new Vector<float>(new float[] {0.0f, 1.0f, 1.5f, 2.0f, -1.0f});
var inputValue = Vector<float>.ZeroVector(5);
IReadOnlyVector<float> result = network.Execute(inputValue);
Console.WriteLine(result);