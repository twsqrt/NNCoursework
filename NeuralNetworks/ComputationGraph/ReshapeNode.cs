using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class ReshapeNode<TInput, TOutput> : Node<TOutput>
    where TInput : ITensor
    where TOutput : ITensor
{
    protected readonly Node<TInput> _input;

    public ReshapeNode(Node<TInput> input, TensorShape3D shape) 
    : base(shape, new INode[]{input})
    {
        _input = input;
    }

    public override void CalculateGradient()
        => _input.ParentGradient = TensorFactory.Create<TInput>(ParentGradient.Data, _input.Shape);

    public override void CalculateValue()
        => _value = TensorFactory.Create<TOutput>(_input.Value.Data, Shape);
    
}
