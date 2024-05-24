using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class MetricNode : Node
{
    private readonly Node _child;
    private readonly ParameterNode _parameter;

    private Vector<float> _difference;

    public MetricNode(Node child, ParameterNode parameter) : base(child.Dimension)
    {
        if(child.Dimension != parameter.Dimension)
            throw new ArgumentException();

        _child = child;
        _parameter = parameter;

        _difference = Vector<float>.CreateZeroVector(Dimension);
    }

    public override void BackpropagateNext(Matrix<float> previouseJacobian)
    {
        _difference.Scale(2.0f);
        Matrix<float> jacobian = _difference.As1DMatrix();

        _child.BackpropagateNext(jacobian.MultiplyRightCached(previouseJacobian));
    }

    public override Vector<float> CalculateValue()
    {
        _difference = _child.CalculateValue() - _parameter.Value;
        return Vector<float>.Create1DVector(_difference.LengthSquared);
    }
}
