using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public abstract class BinaryOperationNode : Node
{
    private readonly Node _lhs;
    private readonly Node _rhs;
    private readonly Matrix<float> _rhsCachedJacobian;
    private Vector<float> _lhsValue;
    private Vector<float> _rhsValue;


    public BinaryOperationNode(Node lhs, Node rhs, int dimension, int graphRootDimension) : base(dimension)
    {
        _lhs = lhs;
        _rhs = rhs;

        _rhsCachedJacobian = Matrix<float>.CreateZeroMatrix(graphRootDimension, Dimension);

        _lhsValue = null;
        _rhsValue = null;
    }

    protected abstract Vector<float> Function(Vector<float> lhs, Vector<float> rhs);
    protected abstract Matrix<float> GetLeftJacobian(Vector<float> lhs, Vector<float> rhs);
    protected abstract Matrix<float> GetRightJacobian(Vector<float> lhs, Vector<float> rhs);

    public override void BackpropagateNext(Matrix<float> previouseJacobian)
    {
        _rhsCachedJacobian.CopyValuesFrom(previouseJacobian);

        _lhs.BackpropagateNext(GetLeftJacobian(_lhsValue, _rhsValue).MultiplyRightCached(previouseJacobian));
        _rhs.BackpropagateNext(GetRightJacobian(_lhsValue, _rhsValue).MultiplyRightCached(_rhsCachedJacobian));
    }

    public override Vector<float> CalculateValue()
    {
        _lhsValue = _lhs.CalculateValue();
        _rhsValue = _rhs.CalculateValue();

        return Function(_lhsValue, _rhsValue);
    }
}
