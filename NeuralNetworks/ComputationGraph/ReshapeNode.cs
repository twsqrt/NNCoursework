using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class ReshapeNode<TInput, TOutput> : Node<TOutput>
    where TInput : ITensor
    where TOutput : ITensor
{
    protected readonly Node<TInput> _input;

    public ReshapeNode(Node<TInput> input, TensorShape shape) 
    : base(shape, new INode[]{input})
    {
        _input = input;
    }

    public override NodeType Type => NodeType.RESHAPE;

    public override void CalculateGradient()
        => _input.Gradient = TensorFactory.Create<TInput>(Gradient.Data, _input.Shape);

    public override void CalculateValue()
        => _value = TensorFactory.Create<TOutput>(_input.Value.Data, Shape);

    protected override void WriteData(BinaryWriter writer)
    {
        writer.Write((byte)_input.Value.Type);
        _input.Export(writer);
    }
}
