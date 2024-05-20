using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class Parameter : Node
{
    private Matrix<float>? _currentJacobian;

    public Matrix<float> CurrentJacobian 
    {
        get 
        {
            if(_currentJacobian is null)
                throw new InvalidOperationException();
            
            return _currentJacobian;
        }
    }

    public Parameter(IReadOnlyVector<float> value) : base(value)
    {
        _currentJacobian = null;
    }

    public void SetValue(IReadOnlyVector<float> value)
    {
        _currentValue.SetValue(value);
    }

    public override IReadOnlyVector<float> UpdateValue()
        => _currentValue;

    public override void BackpropagateNext(Matrix<float> previouseJacobian)
        => _currentJacobian = previouseJacobian;
}