using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public abstract class UnaryOperation : Node
{
    private readonly Node _parameter;

    public int ParameterDimension => _parameter.Dimension;

    public UnaryOperation(Node parameter, int dimension) : base(dimension)
    {
        _parameter = parameter;
    }

    protected abstract Vector<float> Function(IReadOnlyVector<float> at);

    protected abstract IReadOnlyMatrix<float> Jacobian(IReadOnlyVector<float> at);

    public override Vector<float> UpdateValue()
    {
        IReadOnlyVector<float> parameterValue = _parameter.UpdateValue();
        _currentValue.SetValue(Function(parameterValue));

        return _currentValue;
    }

    public override void BackpropagateNext(Matrix<float> previouseJacobian)
    {
        IReadOnlyMatrix<float> jacobian = Jacobian(_parameter.CurrentValue);
        if(jacobian.Size != (Dimension, ParameterDimension))
            throw new InvalidOperationException();

        Matrix<float> nextJacobian = jacobian.MultiplyRightCached(previouseJacobian);

        _parameter.BackpropagateNext(nextJacobian);
    }
}