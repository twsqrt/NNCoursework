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

    protected override Vector<float> Function(Vector<float> parameter)
    {
        (float x, float y) = (parameter[0], parameter[1]);
        return new Vector<float>(new float[] {MathF.Sqrt(x) + y, MathF.Sin(y) + 1.0f});
    }

    protected override Matrix<float> GetJacobian(Vector<float> parameter)
    {
        (float x, float y) = (parameter[0], parameter[1]);
        return new Matrix<float>(2, 2, new float[] {0.5f / MathF.Sqrt(x), 1.0f, 0.0f, MathF.Cos(y)});
    }
}