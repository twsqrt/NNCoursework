using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class TensorToVectorNode : ReshapeNode<Tensor3D, Vector>
{
    public TensorToVectorNode(Node<Tensor3D> input) 
    : base(input, new TensorShape3D(input.Shape.Dimension))
    {
        ParentGradient = Vector.CreateZero(Shape.Dimension);
    }

    public override void CalculateGradient()
        => _input.ParentGradient = new Tensor3D(ParentGradient.Data, _input.Shape);

    public override void CalculateValue()
        => _value = _input.Value.AsVector();
}
