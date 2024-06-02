using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public abstract class ReshapeNode<TInput, TOutput> : Node<TOutput>
    where TInput : ITensor
    where TOutput : ITensor
{
    protected readonly Node<TInput> _input;

    public ReshapeNode(Node<TInput> input, TensorShape3D shape) 
    : base(shape, new INode[]{input})
    {
        _input = input;
    }
}
