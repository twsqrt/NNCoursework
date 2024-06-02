using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class AdditionNode : Node<Vector>
{
    private readonly Node<Vector> _lhs;
    private readonly Node<Vector> _rhs;

    public AdditionNode(Node<Vector> lhs, Node<Vector> rhs) 
    : base(lhs.Shape, new INode[]{lhs, rhs})
    {
        if(lhs.Shape != rhs.Shape)
            throw new ArgumentException();

        _lhs = lhs;
        _rhs = rhs;

        _value = Vector.CreateZero(Shape.Dimension);
        ParentGradient = Vector.CreateZero(Shape.Dimension);
    }

    public override void CalculateGradient()
    {
        _lhs.ParentGradient.CopyValuesFrom(ParentGradient);
        _rhs.ParentGradient = ParentGradient;
    }

    public override void CalculateValue()
        => Vector.Addition(_lhs.Value, _rhs.Value, _value);
}