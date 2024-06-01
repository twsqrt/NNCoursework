namespace NeuralNetworks;

public interface INode
{
    INode[] Parameters { get; }
    void CalculateValue();
}
