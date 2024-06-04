using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class FlattenNode<TInput> : ReshapeNode<TInput, Vector>
    where TInput : ITensor
{
    public FlattenNode(Node<TInput> input) 
    : base(input, new TensorShape(input.Shape.Dimension)) {}
}
