using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public abstract class Node<T> : INode
    where T : ITensor
{
    private readonly INode[] _paramters;
    private readonly TensorShape3D _shape;
    protected T _value;

    public TensorShape3D Shape => _shape;
    public INode[] Parameters => _paramters;
    public T Value => _value;
    internal T ParentGradient;

    public Node(TensorShape3D shape, INode[] parameters)
    {
        _shape = shape;
        _paramters = parameters;

        _value = TensorFactory.CreateZero<T>(shape);
        ParentGradient = TensorFactory.CreateZero<T>(shape);
    }

    public abstract void CalculateValue();
    public abstract void CalculateGradient();
        
}
