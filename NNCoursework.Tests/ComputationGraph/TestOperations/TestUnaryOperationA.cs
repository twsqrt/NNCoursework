using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace ComputationGraph.Tests;

public class TestUnaryOperationA : UnaryOperation
{
    public TestUnaryOperationA(Node parameter) : base(parameter, 2)
    {
        if(parameter.Dimension != 2)
            throw new ArgumentException();
    }

    protected override Vector<float> Function(IReadOnlyVector<float> at)
    {
        (float x, float y) = (at[0], at[1]);
        return new Vector<float>(new float[] {x * x + y * y, x * y + 1.0f});
    }

    protected override IReadOnlyMatrix<float> Jacobian(IReadOnlyVector<float> at)
    {
        (float x, float y) = (at[0], at[1]);
        return new Matrix<float>(2, 2, new float[] {2 * x, 2 * y, y, x});
    }
}