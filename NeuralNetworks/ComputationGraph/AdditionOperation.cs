
using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class AdditionOperation : BinaryOperation
{
    public AdditionOperation(Node leftParameter, Node rightParameter, int dimension) : base(leftParameter, rightParameter, dimension)
    {
        if(leftParameter.Dimension != rightParameter.Dimension
            || leftParameter.Dimension != dimension)
            throw new ArgumentException();
    }

    protected override Vector<float> Function(IReadOnlyVector<float> left, IReadOnlyVector<float> right)
        => Vector<float>.Addition(left, right);

    protected override IReadOnlyMatrix<float> LeftJacobian(IReadOnlyVector<float> left, IReadOnlyVector<float> right)
        => new IdentityMatrix<float>(LeftParameterDimension);

    protected override IReadOnlyMatrix<float> RightJacobian(IReadOnlyVector<float> left, IReadOnlyVector<float> right)
        => new IdentityMatrix<float>(RigthParameterDimension);
}