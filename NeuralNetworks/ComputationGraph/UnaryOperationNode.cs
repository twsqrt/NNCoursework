using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;
public abstract class UnaryOperationNode : Node
{
    private readonly Node _child;
    private readonly Matrix _childCachedJacobian;

    private Vector _childValue;

    public UnaryOperationNode(Node child, int dimension, int graphRootDimension) : base(dimension)
    {
        _child = child;
        _childCachedJacobian = Matrix.CreateZero(graphRootDimension, child.Dimension);

        _childValue = null;
    }

    protected abstract Vector Function(Vector paraemter);
    protected abstract Matrix GetJacobian(Vector parameter);

    public override void BackpropagateNext(Matrix previouseJacobian)
    {
        Matrix jacobian = GetJacobian(_childValue);
        Matrix.Multiply(previouseJacobian, jacobian, _childCachedJacobian);

        _child.BackpropagateNext(_childCachedJacobian);
    }

    public override Vector CalculateValue()
    {
        _childValue = _child.CalculateValue();
        return Function(_childValue);
    }

    public override void Accept(INodeVisitor visitor)
        => throw new InvalidOperationException();
}
