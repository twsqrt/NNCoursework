
using NeuralNetworks.Activation;

namespace NeuralNetworks.ComputationGraph.File;

public class GraphReader
{
    private readonly BinaryReader _reader;
    private readonly Action<int, NodeType, Node> _metadataHandler;

    public GraphReader(BinaryReader reader, Action<int, NodeType, Node> metadataHandler)
    {
        _reader = reader;
        _metadataHandler = metadataHandler;
    }

    public GraphReader(BinaryReader reader) : this(reader, (_, _, _) => {}) {}

    private ParameterNode ReadParameter()
    {
        int dimension = _reader.ReadInt32();

        var data = new float[dimension];
        for(int i = 0; i < dimension; i++)
            data[i] = _reader.ReadSingle();
        
        return ParameterNode.CreateFromArray(data);
    }

    private Node ReadAddition()
    {
        Node lhs = ReadGraph();
        Node rhs = ReadGraph();

        return new AdditionNode(lhs, rhs, 1);
    }

    private Node ReadLayer()
    {
        bool shouldBackpropagateChild = _reader.ReadBoolean();

        int weightsDimension = _reader.ReadInt32();
        var weightsData = new float[weightsDimension];
        for(int i = 0; i < weightsDimension; i++)
            weightsData[i] = _reader.ReadSingle();
        
        ParameterNode weights = ParameterNode.CreateFromArray(weightsData);
        Node child = ReadGraph();

        return new LayerNode(weights, child, 1, shouldBackpropagateChild);
    }

    private Node ReadActivation()
    {
        ActivationType type = (ActivationType) _reader.ReadByte();
        Node child = ReadGraph();

        return ActivationNode.Create(child, type);
    }

    public Node ReadGraph()
    {
        int id = _reader.ReadInt32();
        NodeType type = (NodeType) _reader.ReadByte();

        Node node = type switch
        {
            NodeType.PARAMETER => ReadParameter(),
            NodeType.ADDITION => ReadAddition(),
            NodeType.LAYER => ReadLayer(),
            NodeType.ACTIVATION => ReadActivation(),
            _ => throw new NotImplementedException()
        };

        _metadataHandler(id, type, node);

        return node;
    }
}