using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace ComputationGraph.Tests;

public class TestUnaryOperationB : UnaryOperationNode
{
    public TestUnaryOperationB(Node child) : base(child, 2)
    {
        if(child.Dimension != 2)
            throw new ArgumentException();
    }

    protected override Vector Function(Vector parameter)
    {
        (float x, float y) = (parameter[0], parameter[1]);
        return new Vector(new float[] {MathF.Sqrt(x) + y, MathF.Sin(y) + 1.0f});
    }

    protected override Matrix GetJacobian(Vector parameter)
    {
        (float x, float y) = (parameter[0], parameter[1]);
        return new Matrix(2, 2, new float[] {0.5f / MathF.Sqrt(x), 1.0f, 0.0f, MathF.Cos(y)});
    }
}