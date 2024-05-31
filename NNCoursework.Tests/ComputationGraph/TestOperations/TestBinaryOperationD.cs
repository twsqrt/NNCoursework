using LinearAlgebra;
using NeuralNetworks;
using NeuralNetworks.ComputationGraph;

namespace ComputationGraph.Tests;

public class TestBinaryOperationD : BinaryOperationNode
{
    public TestBinaryOperationD(Node lhs, Node rhs) 
    : base(lhs, rhs, 2)
    {
        if(lhs.Dimension != 3 || rhs.Dimension != 1)
            throw new ArgumentException();
    }

    protected override Vector Function(Vector left, Vector right)
    {
        (float y1, float y2, float y3) = (left[0], left[1], left[2]);
        float x = right.ToNumber();

        return new Vector(new float[] {y1 * x + y2 * y3 + 1.0f / x, 1.0f / (y1 + x) + MathF.Sqrt(y2 * y2 + x + y3)});
    }

    protected override Matrix GetLeftJacobian(Vector left, Vector right)
    {
        (float y1, float y2, float y3) = (left[0], left[1], left[2]);
        float x = right.ToNumber();
        
        var data = new float[]{x, y3, y2, - 1.0f / (y1 + x) / (y1 + x), y2 / MathF.Sqrt(y2 * y2 + x + y3), 0.5f / MathF.Sqrt(y2 * y2 + x + y3)};
        return new Matrix(2, 3, data);
    }

    protected override Matrix GetRightJacobian(Vector left, Vector right)
    {
        (float y1, float y2, float y3) = (left[0], left[1], left[2]);
        float x = right.ToNumber();

        return new Matrix(2, 1, new float[]{y1 -1.0f / x / x, -1.0f / (y1 + x) / (y1 + x) + 0.5f / MathF.Sqrt(y2 * y2 + x + y3)});
    }
}