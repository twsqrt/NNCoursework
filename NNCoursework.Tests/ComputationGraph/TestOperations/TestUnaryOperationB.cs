using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace ComputationGraph.Tests;

public class TestUnaryOperationB : UnaryOperation
{
    public TestUnaryOperationB(Node parameter) : base(parameter, 2)
    {
        if(parameter.Dimension != 2)
            throw new ArgumentException();
    }

    protected override Vector<float> Function(IReadOnlyVector<float> at)
    {
        (float x, float y) = (at[0], at[1]);
        return new Vector<float>(new float[] {MathF.Sqrt(x) + y, MathF.Sin(y) + 1.0f});
    }

    protected override IReadOnlyMatrix<float> Jacobian(IReadOnlyVector<float> at)
    {
        (float x, float y) = (at[0], at[1]);
        return new Matrix<float>(2, 2, new float[] {0.5f / MathF.Sqrt(x), 1.0f, 0.0f, MathF.Cos(y)});
    }
}