using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public abstract class ReshapeNode<TInput, TOutput> : Node<TOutput>
{
    protected readonly Node<TInput> _input;

    public ReshapeNode(Node<TInput> input, TensorShape shape) 
    : base(shape, new INode[]{input})
    {
        _input = input;
    }
}
