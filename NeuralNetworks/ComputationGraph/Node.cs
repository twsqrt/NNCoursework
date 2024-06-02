using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public abstract class Node<T> : INode
    where T : ITensor
{
    private readonly INode[] _paramters;
    private readonly TensorShape _shape;
    protected T _value;

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
    }

    public abstract void CalculateValue();
    public abstract void CalculateGradient();
        
}
