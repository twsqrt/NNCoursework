using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks.Transfer;

public class TransferOperation : UnaryOperation
{
    private readonly ActivationFunction _activationFunction;

    public TransferOperation(Node parameter, ActivationFunction activationFunction) 
    : base(parameter, parameter.Dimension)
    {
        _activationFunction = activationFunction;
    }

    public static TransferOperation Create(Node parameter, ActivationFunctionType type)
        => new TransferOperation(parameter, ActivationFunction.Create(type));

    protected override Vector<float> Function(IReadOnlyVector<float> at)
        => Vector<float>.CreateFromFunction(at.Dimension, i => _activationFunction.Function(at[i]));

    protected override IReadOnlyMatrix<float> Jacobian(IReadOnlyVector<float> at)
    {
        var diagonalElements = new float[Dimension];
        for(int i = 0; i < Dimension; i++)
            diagonalElements[i] = _activationFunction.Function(at[i]);
        
        return new DiagonalMatrix<float>(diagonalElements);
    }
}
