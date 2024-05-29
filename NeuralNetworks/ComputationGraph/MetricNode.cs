using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class MetricNode : Node
{
    private readonly Node _child;
    private readonly ParameterNode _parameter;
    private readonly Matrix<float> _childCachedJacobian;
    private readonly Vector<float> _metricCachedResult;

    private Vector<float> _difference;
    private Matrix<float> _differenceCachedJacobian;

    public MetricNode(Node child, ParameterNode parameter, int graphRootDimension) : base(1)
    {
        if(child.Dimension != parameter.Dimension)
            throw new ArgumentException();

        _child = child;
        _parameter = parameter;

        _childCachedJacobian = Matrix<float>.CreateZeroMatrix(graphRootDimension, child.Dimension);

        _difference = Vector<float>.CreateZeroVector(parameter.Dimension);
        _differenceCachedJacobian = _difference.AsHorizontalMatrix();
        _metricCachedResult = Vector<float>.Create1DVector(0.0f);
    }

    public override void Accept(INodeVisitor visitor)
        => throw new NotImplementedException();

    public override void BackpropagateNext(Matrix<float> previouseJacobian)
    {
        _difference.Scale(2.0f);
        Matrix<float>.Multiply(previouseJacobian, _differenceCachedJacobian, _childCachedJacobian);

        _child.BackpropagateNext(_childCachedJacobian);
    }

    public override Vector<float> CalculateValue()
    {
        Vector<float>.Difference(_child.CalculateValue(), _parameter.Value, _difference);
        _metricCachedResult[0] = _difference.LengthSquared;
        return _metricCachedResult;
    }
}
