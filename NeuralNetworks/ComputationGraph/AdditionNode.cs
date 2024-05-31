using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class AdditionNode : Node
{
    private readonly Node _lhs;
    private readonly Node _rhs;
    private readonly Vector _rhsCachedGradient;
    private readonly Vector _cachedResult;

    public Node LeftNode => _lhs;
    public Node RightNode => _rhs;

    public AdditionNode(Node lhs, Node rhs) : base(lhs.Dimension)
    {
        if(lhs.Dimension != rhs.Dimension)
            throw new ArgumentException();

        _lhs = lhs;
        _rhs = rhs;
        _rhsCachedGradient = Vector.CreateZero(Dimension);
        _cachedResult = Vector.CreateZero(Dimension);
    }

    public override void BackpropagateNext(Vector gradient)
    {
        _rhsCachedGradient.CopyValuesFrom(gradient);

        _lhs.BackpropagateNext(gradient);
        _rhs.BackpropagateNext(_rhsCachedGradient);
    }

    public override Vector CalculateValue()
    {
        Vector.Addition(_lhs.CalculateValue(), _rhs.CalculateValue(), _cachedResult);
        return _cachedResult;
    }

    public override void Accept(INodeVisitor visitor)
        => visitor.Visit(this);
}