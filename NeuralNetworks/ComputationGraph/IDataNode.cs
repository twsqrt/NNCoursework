namespace NeuralNetworks;

public interface IDataNode : INode
{
    float[] Data { get; }
    float[] GradientData { get;}
}
