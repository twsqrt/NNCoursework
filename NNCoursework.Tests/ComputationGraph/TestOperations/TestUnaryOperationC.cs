using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NNCoursework.Tests;

public class TestUnaryOperationC : UnaryOperation
{
    public TestUnaryOperationC(Node parameter) : base(parameter, 1)
    {
        if(parameter.Dimension != 2)
            throw new ArgumentException();
    }

    protected override Vector<float> Function(IReadOnlyVector<float> at)
    {
        (float x, float y) = (at[0], at[1]);
        return new Vector<float>(new float[] {x + y + 1.0f});
    }

    protected override IReadOnlyMatrix<float> Jacobian(IReadOnlyVector<float> at)
        => new Matrix<float>(1, 2, new float[] {1.0f, 1.0f});
}