using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace ComputationGraph.Tests;

public class TestUnaryOperationC : UnaryOperationNode
{
    public TestUnaryOperationC(Node child, int graphRootDimension) : base(child, 1, graphRootDimension)
    {
        if(child.Dimension != 2)
            throw new ArgumentException();
    }

    protected override Vector<float> Function(Vector<float> parameter)
    {
        (float x, float y) = (parameter[0], parameter[1]);
        return new Vector<float>(new float[] {x + y + 1.0f});
    }

    protected override Matrix<float> GetJacobian(Vector<float> parameter)
        => new Matrix<float>(1, 2, new float[] {1.0f, 1.0f});
}