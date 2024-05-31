using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class LossNode : Node
{
    private readonly Node _child;
    private readonly ParameterNode _parameter;
    private readonly Matrix _childCachedJacobian;
    private Vector _difference;
    private Matrix _differenceCachedJacobian;

    public LossNode(Node child, ParameterNode parameter, int graphRootDimension) : base(1)
    {
        if(child.Dimension != parameter.Dimension)
            throw new ArgumentException();

        _child = child;
        _parameter = parameter;

        _childCachedJacobian = Matrix.CreateZero(graphRootDimension, child.Dimension);

        _difference = Vector.CreateZero(parameter.Dimension);
        _differenceCachedJacobian = _difference.AsMatrix(1, _difference.Dimension);
    }

    public override void Accept(INodeVisitor visitor)
        => throw new NotImplementedException();

    public override void BackpropagateNext(Matrix previouseJacobian)
    {
        _difference.Scale(2.0f);
        Matrix.Multiply(previouseJacobian, _differenceCachedJacobian, _childCachedJacobian);
        _child.BackpropagateNext(_childCachedJacobian);
    }

    public override Vector CalculateValue()
    {
        Vector.Difference(_child.CalculateValue(), _parameter.Value, _difference);
        return null;
    }
}
