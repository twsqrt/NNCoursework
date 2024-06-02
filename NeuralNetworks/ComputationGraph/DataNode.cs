using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class DataNode<T> : Node<T>, IDataNode
    where T : ITensor
{
    private readonly float[] _data;
    
    public float[] Data => _data;
    public float[] Gradient => ParentGradient.Data;

    public DataNode(float[] data, T dataWrapper, T parentGradient, TensorShape3D shape)
    : base(shape, new INode[0])
    {
        _data = data;
        _value = dataWrapper;
        ParentGradient = parentGradient;
    }

    public override void CalculateGradient() {}
    public override void CalculateValue() {}
}
