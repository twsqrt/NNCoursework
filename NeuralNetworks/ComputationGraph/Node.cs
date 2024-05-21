using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public abstract class Node
{
    private readonly int _dimension;
    protected readonly Vector<float> _currentValue;

    public int Dimension => _dimension;
    public IReadOnlyVector<float> CurrentValue => _currentValue;

    public Node(IReadOnlyVector<float> value)
    {
        _dimension = value.Dimension;
        _currentValue = Vector<float>.Copy(value);
    }

    public Node(int dimension)
    {
        _dimension = dimension;
        _currentValue = Vector<float>.ZeroVector(dimension);
    }

    public abstract void BackpropagateNext(Matrix<float> previouseJacobian);

    public abstract IReadOnlyVector<float> UpdateValue();
}
