using NeuralNetworks.Activation;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public interface INodeVisitor
{
    void Visit(ParameterNode parameter);
    void Visit(AdditionNode addition);
    void Visit(LayerNode layer);
    void Visit(ActivationNode activation);
}
