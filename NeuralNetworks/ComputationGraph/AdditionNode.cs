using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class AdditionNode<T> : Node<T>
    where T : ITensor
{
    private readonly Node<T> _lhs;
    private readonly Node<T> _rhs;

    public AdditionNode(Node<T> lhs, Node<T> rhs) 
    : base(lhs.Shape, new INode[]{lhs, rhs})
    {
        if(lhs.Shape != rhs.Shape)
            throw new ArgumentException();

        _lhs = lhs;
        _rhs = rhs;
    }

    public override NodeType Type => NodeType.ADDITION;

    public override void CalculateGradient()
    {
        _lhs.Gradient = TensorFactory.Create<T>(Gradient.Data, Shape);
        _rhs.Gradient = Gradient;
    }

    public override void CalculateValue()
    {
        for(int i = 0; i < Shape.Dimension; i++)
            _value.Data[i] = _lhs.Value.Data[i] + _rhs.Value.Data[i];
    }

    protected override void WriteData(BinaryWriter writer)
    {
        _lhs.Export(writer);
        _rhs.Export(writer);
    }
}