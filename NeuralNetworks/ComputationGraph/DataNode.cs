using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class DataNode<T> : Node<T>, IDataNode
    where T : ITensor
{
    public float[] Data => _value.Data;
    public float[] GradientData => Gradient.Data;

    public override NodeType Type => NodeType.DATA;

    public DataNode(float[] data, TensorShape shape)
    : base(shape, new INode[0])
    {
        _value = TensorFactory.Create<T>(data, shape);
    }

    public DataNode(TensorShape shape)
    : base(shape, new INode[0])
    {
        _value = TensorFactory.CreateRandom<T>(shape);
    }

    public override void CalculateGradient() {}

    public override void CalculateValue() {}

    protected override void WriteData(BinaryWriter writer)
    {
        writer.Write(Data.Length);

        for(int i = 0; i < Data.Length; i++)
            writer.Write(Data[i]);
    }
}
