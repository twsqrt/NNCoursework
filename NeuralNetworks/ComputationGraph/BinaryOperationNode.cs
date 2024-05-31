using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public abstract class BinaryOperationNode : Node
{
    private readonly Node _lhs;
    private readonly Node _rhs;
    private readonly Matrix _lhsCachedJacobian;
    private readonly Matrix _rhsCachedJacobian;
    private Vector _lhsValue;
    private Vector _rhsValue;


    public BinaryOperationNode(Node lhs, Node rhs, int dimension, int graphRootDimension) : base(dimension)
    {
        _lhs = lhs;
        _rhs = rhs;

        _lhsCachedJacobian = Matrix.CreateZero(graphRootDimension, lhs.Dimension);
        _rhsCachedJacobian = Matrix.CreateZero(graphRootDimension, rhs.Dimension);

        _lhsValue = null;
        _rhsValue = null;
    }

    protected abstract Vector Function(Vector lhs, Vector rhs);
    protected abstract Matrix GetLeftJacobian(Vector lhs, Vector rhs);
    protected abstract Matrix GetRightJacobian(Vector lhs, Vector rhs);

    public override void BackpropagateNext(Matrix previouseJacobian)
    {
        Matrix.Multiply(previouseJacobian, GetLeftJacobian(_lhsValue, _rhsValue), _lhsCachedJacobian);
        Matrix.Multiply(previouseJacobian, GetRightJacobian(_lhsValue, _rhsValue), _rhsCachedJacobian);

        _lhs.BackpropagateNext(_lhsCachedJacobian);
        _rhs.BackpropagateNext(_rhsCachedJacobian);;
    }

    public override Vector CalculateValue()
    {
        _lhsValue = _lhs.CalculateValue();
        _rhsValue = _rhs.CalculateValue();

        return Function(_lhsValue, _rhsValue);
    }

    public override void Accept(INodeVisitor visitor)
        => throw new InvalidOperationException();
}
