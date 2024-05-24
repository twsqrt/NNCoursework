using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public abstract class Node
{
    private readonly int _dimension;

    public int Dimension => _dimension;

    public Node(int dimension)
    {
        _dimension = dimension;
    }

    public abstract void BackpropagateNext(Matrix<float> previouseJacobian);

    public abstract Vector<float> CalculateValue();

    public void Backpropagate()
        => BackpropagateNext(Matrix<float>.CreateIdentityMatrix(_dimension));
}
