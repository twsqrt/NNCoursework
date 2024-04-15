using LinearAlgebra;

namespace NeuralNetworks.Transfer;

public class TransferFunction
{
    private readonly Func<double, double> _activationFunction;

    public TransferFunction(Func<double, double> activationFunction)
    {
        _activationFunction = activationFunction;
    }

    public static TransferFunction Create(TransferFunctionType functionType)
        => new TransferFunction(ActivationFunctions.Get(functionType));

    public Vector<double> Execute(Vector<double> vector)
        => Vector<double>.CreateFromFunction(vector.Length, i => _activationFunction(vector[i]));
}
