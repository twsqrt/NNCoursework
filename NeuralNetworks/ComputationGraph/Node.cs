using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public abstract class Node
{
    private static int _idCounter = 0;

    private readonly int _id;
    private readonly int _dimension;

    public int ID => _id;
    public int Dimension => _dimension;

    public Node(int dimension)
    {
        _dimension = dimension;

        _id = _idCounter;
        _idCounter++;
    }

    public abstract void BackpropagateNext(Vector gradient);
    public abstract Vector CalculateValue();
    public abstract void Accept(INodeVisitor visitor);
}
