using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public abstract class Node<T>
{
    private static int _idCounter = 0;

    private readonly int _id;
    private readonly TensorShape _shape;

    public int ID => _id;
    public TensorShape Shape => _shape;

    public Node(TensorShape shape)
    {
        _shape = shape;
        _id = _idCounter;

        _idCounter++;
    }

    public abstract void BackpropagateNext(Vector gradient);
    public abstract T CalculateValue();
    public abstract void Accept(INodeVisitor visitor);
}
