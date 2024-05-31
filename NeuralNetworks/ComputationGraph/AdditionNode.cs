using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class AdditionNode : Node
{
    private readonly Node _lhs;
    private readonly Node _rhs;
    private readonly Matrix _rhsCachedJacobian;
    private readonly Vector _cachedResult;

    public Node LeftNode => _lhs;
    public Node RightNode => _rhs;

    public AdditionNode(Node lhs, Node rhs, int graphRootDimension) : base(lhs.Dimension)
    {
        if(lhs.Dimension != rhs.Dimension)
            throw new ArgumentException();

        _lhs = lhs;
        _rhs = rhs;
        _rhsCachedJacobian = Matrix.CreateZero(graphRootDimension, Dimension);
        _cachedResult = Vector.CreateZero(Dimension);
    }

    public override void BackpropagateNext(Matrix previouseJacobian)
    {
        _rhsCachedJacobian.CopyValuesFrom(previouseJacobian);

        _lhs.BackpropagateNext(previouseJacobian);
        _rhs.BackpropagateNext(_rhsCachedJacobian);
    }

    public override Vector CalculateValue()
    {
        Vector.Addition(_lhs.CalculateValue(), _rhs.CalculateValue(), _cachedResult);
        return _cachedResult;
    }

    public override void Accept(INodeVisitor visitor)
        => visitor.Visit(this);
}