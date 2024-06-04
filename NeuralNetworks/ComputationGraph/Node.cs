using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public abstract class Node<T> : INode
    where T : ITensor
{
    private static int _idCounter = 0;

    private readonly int _id;
    private readonly INode[] _paramters;
    private readonly TensorShape _shape;
    protected T _value;

    public int ID => _id;
    public abstract NodeType Type { get; }
    public TensorShape Shape => _shape;
    public INode[] Parameters => _paramters;
    public T Value => _value;
    internal T Gradient;


    public Node(TensorShape shape, INode[] parameters)
    {
        _shape = shape;
        _paramters = parameters;

        _value = TensorFactory.CreateZero<T>(shape);
        Gradient = TensorFactory.CreateZero<T>(shape);
       
        _id = _idCounter;
        _idCounter++;
    }

    public abstract void CalculateValue();
    public abstract void CalculateGradient();

    private void WriteHeader(BinaryWriter writer)
    {
        writer.Write(_id);
        writer.Write((byte)Type);
        writer.Write((byte)_value.Type);
        writer.Write(_shape.Height);
        writer.Write(_shape.Width);
        writer.Write(_shape.Depth);
    }

    protected abstract void WriteData(BinaryWriter writer);
    
    public void Export(BinaryWriter writer)
    {
        WriteHeader(writer);
        WriteData(writer);
    }
}
