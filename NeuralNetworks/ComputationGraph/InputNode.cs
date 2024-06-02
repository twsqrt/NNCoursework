using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class VectorInputNode : Node<Vector>
{
    public new Vector Value
    {
        get => _value;
        set => _value = value;
    }

    public VectorInputNode(int dimension)
    : base(new TensorShape3D(dimension), new INode[0])
    {
        ParentGradient = Vector.CreateZero(dimension);
    }

    public override void CalculateGradient() {}

    public override void CalculateValue() {}
}
