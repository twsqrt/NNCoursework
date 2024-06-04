using System.Runtime.CompilerServices;
using LinearAlgebra;
using NeuralNetworks.Activation;

namespace NeuralNetworks.ComputationGraph.Reader;

public readonly struct NodeHeader
{
    public readonly int ID;
    public readonly NodeType NodeType;
    public readonly TensorType ValueType;
    public readonly TensorShape Shape;

    public NodeHeader(int id, NodeType nodeType, TensorType valueType, TensorShape shape)
    {
        ID = id;
        NodeType = nodeType;
        ValueType = valueType;
        Shape = shape;
    }
}

public class GraphReader
{
    private readonly BinaryReader _reader;
    private readonly Action<NodeHeader, INode> _nodeHandler;

    public GraphReader(BinaryReader reader, Action<NodeHeader, INode> nodeHandler)
    {
        _reader = reader;
        _nodeHandler = nodeHandler;
    }

    private NodeHeader ReadHeader()
    {
        int id = _reader.ReadInt32();
        NodeType type = (NodeType) _reader.ReadByte();
        TensorType valueType = (TensorType) _reader.ReadByte();
        
        int height = _reader.ReadInt32();
        int width = _reader.ReadInt32();
        int depth = _reader.ReadInt32();
        TensorShape shpae = new TensorShape(height, width, depth);

        return new NodeHeader(id, type, valueType, shpae);
    }

    private DataNode<T> ReadDataNode<T>(NodeHeader header)
        where T : ITensor
    {
        int lenght = _reader.ReadInt32();

        var data = new float[lenght];
        for(int i = 0; i < lenght; i++)
            data[i] = _reader.ReadSingle();
         
        return new DataNode<T>(data, header.Shape);
    }

    private AdditionNode<T> ReadAdditionNode<T>()
        where T : ITensor
    {
        Node<T> lhs = ReadNode<T>();
        Node<T> rhs = ReadNode<T>();

        return new AdditionNode<T>(lhs, rhs);
    }

    private LayerNode ReadLayerNode()
    {
        bool shouldBackpropagateChild = _reader.ReadBoolean();

        Node<Vector> child = ReadNode<Vector>();
        Node<Matrix> weights = ReadNode<Matrix>();

        return new LayerNode(weights, child, shouldBackpropagateChild);
    }

    private ActivationNode<T> ReadActivationNode<T>()
        where T : ITensor
    {
        ActivationType type = (ActivationType) _reader.ReadByte();
        Node<T> child = ReadNode<T>();

        return ActivationNode<T>.Create(child, type);
    }

    private ReshapeNode<TInput, TOutput> CreateReshapeNode<TInput, TOutput>(NodeHeader header)
        where TInput : ITensor where TOutput : ITensor
        => new ReshapeNode<TInput, TOutput>(ReadNode<TInput>(), header.Shape);


    private Node<T> ReadReshapeNode<T>(NodeHeader header)
        where T : ITensor
    {
        TensorType inputType = (TensorType) _reader.ReadByte();

        return inputType switch
        {
            TensorType.VECTOR => CreateReshapeNode<Vector, T>(header),
            TensorType.MATRIX => CreateReshapeNode<Matrix, T>(header),
            TensorType.TENSOR3D => CreateReshapeNode<Tensor3D, T>(header),
            _ => throw new NotImplementedException(),
        };
    }

    private Convolution2DNode ReadConvolution2DNode()
    {
        bool shouldBackpropagateChild = _reader.ReadBoolean();
        Node<Matrix> child = ReadNode<Matrix>();
        Node<Tensor3D> kernel = ReadNode<Tensor3D>();

        return new Convolution2DNode(child, kernel, shouldBackpropagateChild);
    }

    private MaxPool2DNode ReadMaxPool2DNode()
    {
        int kernelHeight = _reader.ReadInt32();
        int kernelWidth = _reader.ReadInt32();
        Node<Tensor3D> child = ReadNode<Tensor3D>();

        return new MaxPool2DNode(child, kernelHeight, kernelWidth);
    }


    public Node<T> ReadNode<T>()
        where T : ITensor
    {
        NodeHeader header = ReadHeader();

        Node<T> node = header.NodeType switch
        {
            NodeType.LOSS => throw new InvalidOperationException(),
            NodeType.DATA => ReadDataNode<T>(header),
            NodeType.INPUT => new VectorInputNode(header.Shape.Dimension) as Node<T>,
            NodeType.ADDITION => ReadAdditionNode<T>(),
            NodeType.LAYER => ReadLayerNode() as Node<T>,
            NodeType.ACTIVATION => ReadActivationNode<T>(),
            NodeType.SOFTMAX => new SoftMaxNode(ReadNode<Vector>()) as Node<T>,
            NodeType.RESHAPE => ReadReshapeNode<T>(header),
            NodeType.CONVOLUTION2D => ReadConvolution2DNode() as Node<T>,
            NodeType.MAXPOOL2D => ReadMaxPool2DNode() as Node<T>,
            _ => throw new NotImplementedException(),
        };

        _nodeHandler(header, node);

        return node;
    }
}
