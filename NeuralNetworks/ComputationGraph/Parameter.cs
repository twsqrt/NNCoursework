using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class Parameter : Node
{
    private Matrix<float> _currentJacobian;

    public Matrix<float> CurrentJacobian => _currentJacobian;

    private Parameter(IReadOnlyVector<float> value) : base(value)
    {
        _currentJacobian = null;
    }

    public static Parameter CreateFrom(IReadOnlyVector<float> value)
        => new Parameter(value);
    
    public static Parameter CreateZero(int dimension)
        => new Parameter(Vector<float>.ZeroVector(dimension));

    public void SetValue(IReadOnlyVector<float> value)
    {
        _currentValue.SetValue(value);
    }

    public override IReadOnlyVector<float> UpdateValue()
        => _currentValue;

    public override void BackpropagateNext(Matrix<float> previouseJacobian)
        => _currentJacobian = previouseJacobian;
}