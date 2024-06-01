using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class VectorToTensor : ReshapeNode<Vector, Tensor>
{
    public VectorToTensor(Node<Vector> input, TensorShape shape) 
    : base(input, shape) {}

    public override void BackpropagateNext(Tensor gradient)
        => _input.BackpropagateNext(gradient.AsVector());

    public override void CalculateValue()
    {
        _value = new Tensor(_input.Value.Data, Shape);
    }
}
