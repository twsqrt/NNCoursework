using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class MetricNode : Node
{
    private readonly Node _child;
    private readonly ParameterNode _parameter;
    private readonly Matrix<float> _childCachedJacobian;

    private Vector<float> _difference;

    public MetricNode(Node child, ParameterNode parameter, int graphRootDimension) : base(1)
    {
        if(child.Dimension != parameter.Dimension)
            throw new ArgumentException();

        _child = child;
        _parameter = parameter;

        _childCachedJacobian = Matrix<float>.CreateZeroMatrix(graphRootDimension, child.Dimension);

        _difference = null;
    }

    public override void BackpropagateNext(Matrix<float> previouseJacobian)
    {
        _difference.Scale(2.0f);
        Matrix<float> jacobian = _difference.AsHorizontalMatrix();
        Matrix<float>.Multiply(previouseJacobian, jacobian, _childCachedJacobian);

        _child.BackpropagateNext(_childCachedJacobian);
    }

    public override Vector<float> CalculateValue()
    {
        _difference = _child.CalculateValue() - _parameter.Value;
        return Vector<float>.Create1DVector(_difference.LengthSquared);
    }
}
