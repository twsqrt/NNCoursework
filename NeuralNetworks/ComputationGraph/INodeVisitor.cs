using NeuralNetworks.Activation;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public interface INodeVisitor
{
    void Visit(DataNode data);
    void Visit(AdditionNode addition);
    void Visit(LayerNode layer);
    void Visit(ActivationNode activation);
}
