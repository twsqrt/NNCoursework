using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public abstract class ReshapeNode<TInput, TOutput> : Node<TOutput>
{
    protected readonly Node<TInput> _input;

    public ReshapeNode(Node<TInput> input, TensorShape shape) 
    : base(shape)
    {
        _input = input;
    }

    public override void Accept(INodeVisitor visitor)
    {
        throw new NotImplementedException();
    }
}
