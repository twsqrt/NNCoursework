using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class SquareMetric : BinaryOperation
{
    public SquareMetric(Node leftParameter, Node rightParameter) : base(leftParameter, rightParameter, 1)
    {
        if(leftParameter.Dimension != rightParameter.Dimension)
            throw new ArgumentException();
    }

    protected override Vector<float> Function(IReadOnlyVector<float> left, IReadOnlyVector<float> right)
    {
        float index = Vector<float>.Difference(left, right).LengthSquared;
        return new Vector<float>(new float[] {index});
    }

    protected override IReadOnlyMatrix<float> LeftJacobian(IReadOnlyVector<float> left, IReadOnlyVector<float> right)
    {
        Vector<float> difference = Vector<float>.Difference(left, right);
        difference.MultiplyByScalar(2.0f);
        return difference.ToMatrixCached(1, difference.Dimension);
    }

    protected override IReadOnlyMatrix<float> RightJacobian(IReadOnlyVector<float> left, IReadOnlyVector<float> right)
        => LeftJacobian(right, left);
}
