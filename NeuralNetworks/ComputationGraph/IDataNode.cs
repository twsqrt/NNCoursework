namespace NeuralNetworks;

public interface IDataNode
{
    float[] Data { get; }
    float[] GradientData { get;}
}
