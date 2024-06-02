using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class VectorToTensor : ReshapeNode<Vector, Tensor3D>
{
    public VectorToTensor(Node<Vector> input, TensorShape3D shape) 
    : base(input, shape)
    {
        ParentGradient = Tensor3D.CreateZero(shape);
    }

    public override void CalculateGradient()
        => _input.ParentGradient = new Vector(ParentGradient.Data);

    public override void CalculateValue()
        => _value = new Tensor3D(_input.Value.Data, Shape);
}
