using System.Net.Http.Headers;
using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace ComputationGraph.Tests;

public class TestBinaryOperationD : BinaryOperation
{
    public TestBinaryOperationD(Node leftParameter, Node rightParameter) : base(leftParameter, rightParameter, 2)
    {
        if(leftParameter.Dimension != 3
            || rightParameter.Dimension != 1)
            throw new ArgumentException();
    }

    protected override Vector<float> Function(IReadOnlyVector<float> left, IReadOnlyVector<float> right)
    {
        (float y1, float y2, float y3) = (left[0], left[1], left[2]);
        float x = right.ToNumber();

        return new Vector<float>(new float[] {y1 * x + y2 * y3 + 1.0f / x, 1.0f / (y1 + x) + MathF.Sqrt(y2 * y2 + x + y3)});
    }

    protected override IReadOnlyMatrix<float> LeftJacobian(IReadOnlyVector<float> left, IReadOnlyVector<float> right)
    {
        (float y1, float y2, float y3) = (left[0], left[1], left[2]);
        float x = right.ToNumber();
        
        var data = new float[]{x, y3, y2, - 1.0f / (y1 + x) / (y1 + x), y2 / MathF.Sqrt(y2 * y2 + x + y3), 0.5f / MathF.Sqrt(y2 * y2 + x + y3)};
        return new Matrix<float>(2, 3, data);
    }

    protected override IReadOnlyMatrix<float> RightJacobian(IReadOnlyVector<float> left, IReadOnlyVector<float> right)
    {
        (float y1, float y2, float y3) = (left[0], left[1], left[2]);
        float x = right.ToNumber();

        return new Matrix<float>(2, 1, new float[]{y1 -1.0f / x / x, -1.0f / (y1 + x) / (y1 + x) + 0.5f / MathF.Sqrt(y2 * y2 + x + y3)});
    }
}