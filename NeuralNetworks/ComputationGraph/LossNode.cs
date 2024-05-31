using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class LossNode : Node
{
    private readonly ParameterNode _parameter;
    private readonly Node _child;
    private readonly Vector _childCachedGradient;
    private Vector _difference;
    private Matrix _differenceCachedJacobian;

    public LossNode(Node child, ParameterNode parameter) : base(1)
    {
        if(child.Dimension != parameter.Dimension)
            throw new ArgumentException();

        _child = child;
        _parameter = parameter;

        _childCachedGradient = Vector.CreateZero(child.Dimension);

        _difference = Vector.CreateZero(parameter.Dimension);
        _differenceCachedJacobian = _difference.AsMatrix(1, _difference.Dimension);
    }

    public override void Accept(INodeVisitor visitor)
        => throw new NotImplementedException();

    public override void BackpropagateNext(Vector gradient)
        => throw new InvalidOperationException();

    public override Vector CalculateValue()
    {
        Vector.Difference(_child.CalculateValue(), _parameter.Value, _difference);
        return Vector.Create1D(_difference.LengthSquared);
    }

    public void Backpropagate()
    {
        _difference.Scale(2.0f);
        _child.BackpropagateNext(_difference);
    }
}
