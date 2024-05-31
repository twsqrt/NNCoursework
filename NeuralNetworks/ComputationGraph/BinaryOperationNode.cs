using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public abstract class BinaryOperationNode : Node
{
    private readonly Node _lhs;
    private readonly Node _rhs;
    private readonly Vector _lhsCachedGradient;
    private readonly Vector _rhsCachedGradient;
    private Vector _lhsValue;
    private Vector _rhsValue;


    public BinaryOperationNode(Node lhs, Node rhs, int dimension) : base(dimension)
    {
        _lhs = lhs;
        _rhs = rhs;

        _lhsCachedGradient = Vector.CreateZero(lhs.Dimension);
        _rhsCachedGradient = Vector.CreateZero(rhs.Dimension);
    }

    protected abstract Vector Function(Vector lhs, Vector rhs);
    protected abstract Matrix GetLeftJacobian(Vector lhs, Vector rhs);
    protected abstract Matrix GetRightJacobian(Vector lhs, Vector rhs);

    public override void BackpropagateNext(Vector gradient)
    {
        Matrix.Multiply(gradient, GetLeftJacobian(_lhsValue, _rhsValue), _lhsCachedGradient);
        Matrix.Multiply(gradient, GetRightJacobian(_lhsValue, _rhsValue), _rhsCachedGradient);

        _lhs.BackpropagateNext(_lhsCachedGradient);
        _rhs.BackpropagateNext(_rhsCachedGradient);
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
