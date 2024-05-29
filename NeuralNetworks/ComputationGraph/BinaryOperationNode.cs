using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public abstract class BinaryOperationNode : Node
{
    private readonly Node _lhs;
    private readonly Node _rhs;
    private readonly Matrix<float> _lhsCachedJacobian;
    private readonly Matrix<float> _rhsCachedJacobian;
    private Vector<float> _lhsValue;
    private Vector<float> _rhsValue;


    public BinaryOperationNode(Node lhs, Node rhs, int dimension, int graphRootDimension) : base(dimension)
    {
        _lhs = lhs;
        _rhs = rhs;

        _lhsCachedJacobian = Matrix<float>.CreateZeroMatrix(graphRootDimension, lhs.Dimension);
        _rhsCachedJacobian = Matrix<float>.CreateZeroMatrix(graphRootDimension, rhs.Dimension);

        _lhsValue = null;
        _rhsValue = null;
    }

    protected abstract Vector<float> Function(Vector<float> lhs, Vector<float> rhs);
    protected abstract Matrix<float> GetLeftJacobian(Vector<float> lhs, Vector<float> rhs);
    protected abstract Matrix<float> GetRightJacobian(Vector<float> lhs, Vector<float> rhs);

    public override void BackpropagateNext(Matrix<float> previouseJacobian)
    {
        Matrix<float>.Multiply(previouseJacobian, GetLeftJacobian(_lhsValue, _rhsValue), _lhsCachedJacobian);
        Matrix<float>.Multiply(previouseJacobian, GetRightJacobian(_lhsValue, _rhsValue), _rhsCachedJacobian);

        _lhs.BackpropagateNext(_lhsCachedJacobian);
        _rhs.BackpropagateNext(_rhsCachedJacobian);;
    }

    public override Vector<float> CalculateValue()
    {
        _lhsValue = _lhs.CalculateValue();
        _rhsValue = _rhs.CalculateValue();

        return Function(_lhsValue, _rhsValue);
    }

    public override void Accept(INodeVisitor visitor)
        => throw new InvalidOperationException();
}
