using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class AdditionNode : Node<Vector>
{
    private readonly Node<Vector> _lhs;
    private readonly Node<Vector> _rhs;
    private readonly Vector _rhsCachedGradient;
    private readonly Vector _cachedResult;

    public Node<Vector> LeftNode => _lhs;
    public Node<Vector> RightNode => _rhs;

    public AdditionNode(Node<Vector> lhs, Node<Vector> rhs) : base(lhs.Shape)
    {
        if(lhs.Shape != rhs.Shape)
            throw new ArgumentException();

        _lhs = lhs;
        _rhs = rhs;

        _rhsCachedGradient = Vector.CreateZero(Shape.Height);
        _cachedResult = Vector.CreateZero(Shape.Height);
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