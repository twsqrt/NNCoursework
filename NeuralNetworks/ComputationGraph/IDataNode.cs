namespace NeuralNetworks;

public interface IDataNode
{
    float[] Data { get; }
    float[] Gradient { get;}
}
