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
    }

    public override void CalculateGradient()
    {
        _lhs.Gradient.CopyValuesFrom(Gradient);
        _rhs.Gradient = Gradient;
    }

    public override void CalculateValue()
        => Vector.Addition(_lhs.Value, _rhs.Value, _value);
}