using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class DataNode<T> : Node<T>, IDataNode
    where T : ITensor
{
    public float[] Data => _value.Data;
    public float[] Gradient => ParentGradient.Data;

    public DataNode(TensorShape3D shape)
    : base(shape, new INode[0])
    {
        _value = TensorFactory.CreateRandom<T>(shape);
    }

    public override void CalculateGradient() {}
    public override void CalculateValue() {}
}
