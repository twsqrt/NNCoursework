using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;
public abstract class UnaryOperationNode : Node
{
    private readonly Node _child;
    private readonly Matrix<float> _childCachedJacobian;

    private Vector<float> _childValue;

    public UnaryOperationNode(Node child, int dimension, int graphRootDimension) : base(dimension)
    {
        _child = child;
        _childCachedJacobian = Matrix<float>.CreateZeroMatrix(graphRootDimension, child.Dimension);

        _childValue = null;
    }

    protected abstract Vector<float> Function(Vector<float> paraemter);
    protected abstract Matrix<float> GetJacobian(Vector<float> parameter);

    public override void BackpropagateNext(Matrix<float> previouseJacobian)
    {
        Matrix<float> jacobian = GetJacobian(_childValue);
        Matrix<float>.Multiply(previouseJacobian, jacobian, _childCachedJacobian);

        _child.BackpropagateNext(_childCachedJacobian);
    }

    public override Vector<float> CalculateValue()
    {
        _childValue = _child.CalculateValue();
        return Function(_childValue);
    }

    public override void Accept(INodeVisitor visitor)
        => throw new InvalidOperationException();
}
