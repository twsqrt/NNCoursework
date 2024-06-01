using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class AdditionNode : Node<Vector>
{
    private readonly Node<Vector> _lhs;
    private readonly Node<Vector> _rhs;
    private readonly Vector _rhsCachedGradient;

    public AdditionNode(Node<Vector> lhs, Node<Vector> rhs) 
    : base(lhs.Shape, new INode[]{lhs, rhs})
    {
        if(lhs.Shape != rhs.Shape)
            throw new ArgumentException();

        _lhs = lhs;
        _rhs = rhs;

        _rhsCachedGradient = Vector.CreateZero(Shape.Height);
        _value = Vector.CreateZero(Shape.Height);
    }

    public override void BackpropagateNext(Vector gradient)
    {
        _rhsCachedGradient.CopyValuesFrom(gradient);

        _lhs.BackpropagateNext(gradient);
        _rhs.BackpropagateNext(_rhsCachedGradient);
    }

    public override void CalculateValue()
        => Vector.Addition(_lhs.Value, _rhs.Value, _value);
}