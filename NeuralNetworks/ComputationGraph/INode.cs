using LinearAlgebra;

namespace NeuralNetworks;

public interface INode
{
    INode[] Parameters { get; }
    void CalculateGradient();
    void CalculateValue();
}
