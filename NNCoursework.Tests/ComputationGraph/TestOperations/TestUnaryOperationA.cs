using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace ComputationGraph.Tests;

public class TestUnaryOperationA : UnaryOperationNode
{
    public TestUnaryOperationA(Node child) : base(child, 2)
    {
        if(child.Dimension != 2)
            throw new ArgumentException();
    }

    protected override Vector Function(Vector parameter)
    {
        (float x, float y) = (parameter[0], parameter[1]);
        return new Vector(new float[] {x * x + y * y, x * y + 1.0f});
    }

    protected override Matrix GetJacobian(Vector parameter)
    {
        (float x, float y) = (parameter[0], parameter[1]);
        return new Matrix(2, 2, new float[] {2 * x, 2 * y, y, x});
    }
}