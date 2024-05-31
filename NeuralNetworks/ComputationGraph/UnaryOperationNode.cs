using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;
public abstract class UnaryOperationNode : Node
{
    private readonly Node _child;
    private readonly Vector _childCachedGradient;

    private Vector _childValue;

    public UnaryOperationNode(Node child, int dimension) : base(dimension)
    {
        _child = child;
        _childCachedGradient = Vector.CreateZero(child.Dimension);

        _childValue = null;
    }

    protected abstract Vector Function(Vector paraemter);
    protected abstract Matrix GetJacobian(Vector parameter);

    public override void BackpropagateNext(Vector gradient)
    {
        Matrix jacobian = GetJacobian(_childValue);
        Matrix.Multiply(gradient, jacobian, _childCachedGradient);
        _child.BackpropagateNext(_childCachedGradient);
    }

    public override Vector CalculateValue()
    {
        _childValue = _child.CalculateValue();
        return Function(_childValue);
    }

    public override void Accept(INodeVisitor visitor)
        => throw new InvalidOperationException();
}
