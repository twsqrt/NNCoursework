using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks.Transfer;

public class TransferOperation : UnaryOperation
{
    private readonly ActivationFunction _activationFunction;

    public TransferOperation(Node parameter, ActivationFunction activationFunction, int dimension) 
        : base(parameter, dimension)
    {
        if(parameter.Dimension != dimension)
            throw new ArgumentException();

        _activationFunction = activationFunction;
    }

    protected override Vector<float> Function(IReadOnlyVector<float> at)
        => Vector<float>.CreateFromFunction(at.Length, i => _activationFunction.Function(at[i]));

    protected override IReadOnlyMatrix<float> Jacobian(IReadOnlyVector<float> at)
    {
        var diagonalElements = new float[Dimension];
        for(int i = 0; i < Dimension; i++)
            diagonalElements[i] = _activationFunction.Function(at[i]);
        
        return new DiagonalMatrix<float>(diagonalElements);
    }
}
