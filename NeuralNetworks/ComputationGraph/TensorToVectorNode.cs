using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class TensorToVectorNode : ReshapeNode<Tensor, Vector>
{
    public TensorToVectorNode(Node<Tensor> input) 
    : base(input, new TensorShape(input.Shape.Dimension)) {}

    public override void BackpropagateNext(Vector gradient)
        => _input.BackpropagateNext(new Tensor(gradient.Data, _input.Shape));

    public override Vector CalculateValue()
        => _input.CalculateValue().AsVector();
}
