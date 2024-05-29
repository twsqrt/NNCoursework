using NeuralNetworks.Activation;

namespace NeuralNetworks.ComputationGraph.File;

public class GraphWriter : INodeVisitor
{
    private readonly BinaryWriter _writer;

    public GraphWriter(BinaryWriter writer)
    {
        _writer = writer;
    }

    private void WriteHeader(int id, NodeType type)
    {
        _writer.Write(id);
        _writer.Write((byte)type);
    }

    public void Visit(ParameterNode parameter)
    {
        WriteHeader(parameter.ID, NodeType.PARAMETER);

        _writer.Write(parameter.Dimension);
        for(int i = 0; i < parameter.Dimension; i++)
            _writer.Write(parameter.Value[i]);
    }

    public void Visit(AdditionNode addition)
    {
        WriteHeader(addition.ID, NodeType.ADDITION);

        addition.LeftNode.Accept(this);
        addition.RightNode.Accept(this);
    }

    public void Visit(LayerNode layer)
    {
        WriteHeader(layer.ID, NodeType.LAYER);

        _writer.Write(layer.ShouldBackpropagateChild);

        ParameterNode weights = layer.Weights;
        _writer.Write(weights.Dimension);
        for(int i = 0; i < weights.Dimension; i++)
            _writer.Write(weights.Value[i]);

        layer.Child.Accept(this);
    }

    public void Visit(ActivationNode activation)
    {
        if(activation.Type == ActivationType.CUSTOM)
            throw new InvalidOperationException();

        WriteHeader(activation.ID, NodeType.ACTIVATION);

        _writer.Write((byte)activation.Type);
        activation.Child.Accept(this);
    }
}
