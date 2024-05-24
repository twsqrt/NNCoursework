using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace ComputationGraph.Tests;

public class TestUnaryOperationA : UnaryOperationNode
{
    public TestUnaryOperationA(Node child, int graphRootDimension) : base(child, 2, graphRootDimension)
    {
        if(child.Dimension != 2)
            throw new ArgumentException();
    }

    protected override Vector<float> Function(Vector<float> parameter)
    {
        (float x, float y) = (parameter[0], parameter[1]);
        return new Vector<float>(new float[] {x * x + y * y, x * y + 1.0f});
    }

    protected override Matrix<float> GetJacobian(Vector<float> parameter)
    {
        (float x, float y) = (parameter[0], parameter[1]);
        return new Matrix<float>(2, 2, new float[] {2 * x, 2 * y, y, x});
    }
}