using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public abstract class Node<T> : INode
{
    private static int _idCounter = 0;

    private readonly INode[] _paramters;
    private readonly int _id;
    private readonly TensorShape _shape;
    protected T _value;

    public int ID => _id;
    public TensorShape Shape => _shape;
    public T Value => _value;
    public INode[] Parameters => _paramters;

    public Node(TensorShape shape, INode[] parameters)
    {
        _shape = shape;
        _paramters = parameters;
        _id = _idCounter;

        _idCounter++;
    }

    public abstract void CalculateValue();
    public abstract void BackpropagateNext(T gradient);
}
