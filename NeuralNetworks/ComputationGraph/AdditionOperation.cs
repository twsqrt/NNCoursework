
using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class AdditionOperation : BinaryOperation
{
    public AdditionOperation(Node leftParameter, Node rightParameter) 
    : base(leftParameter, rightParameter, leftParameter.Dimension)
    {
        if(leftParameter.Dimension != rightParameter.Dimension)
            throw new ArgumentException();
    }

    protected override Vector<float> Function(IReadOnlyVector<float> left, IReadOnlyVector<float> right)
        => Vector<float>.Addition(left, right);

    protected override IReadOnlyMatrix<float> LeftJacobian(IReadOnlyVector<float> left, IReadOnlyVector<float> right)
        => new CreateIdentityMatrix<float>(LeftParameterDimension);

    protected override IReadOnlyMatrix<float> RightJacobian(IReadOnlyVector<float> left, IReadOnlyVector<float> right)
        => new CreateIdentityMatrix<float>(RigthParameterDimension);
}