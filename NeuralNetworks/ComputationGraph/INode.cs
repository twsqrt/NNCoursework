namespace NeuralNetworks;

public interface INode
{
    INode[] Parameters { get; }
    int ID { get; }
    NodeType Type { get; }
    void CalculateGradient();
    void CalculateValue();
    void Export(BinaryWriter writer);
}
