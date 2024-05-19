using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public abstract class BinaryOperation : Node
{
    private readonly Node _leftParameter;
    private readonly Node _rightParameter;

    public int LeftParameterDimension => _leftParameter.Dimension;
    public int RigthParameterDimension => _rightParameter.Dimension;

    public BinaryOperation(Node leftParameter, Node rightParameter, int dimension) : base(dimension)
    {
        _leftParameter = leftParameter;
        _rightParameter = rightParameter;
    }

    protected abstract Vector<float> Function(IReadOnlyVector<float> left, IReadOnlyVector<float> right);

    protected abstract IReadOnlyMatrix<float> LeftJacobian(IReadOnlyVector<float> left, IReadOnlyVector<float> right);

    protected abstract IReadOnlyMatrix<float> RightJacobian(IReadOnlyVector<float> left, IReadOnlyVector<float> right);

    public override IReadOnlyVector<float> UpdateValue()
    {
        IReadOnlyVector<float> leftParameterValue = _leftParameter.UpdateValue();
        IReadOnlyVector<float> rightParameterValue = _rightParameter.UpdateValue();
        _currentValue.SetValue(Function(leftParameterValue, rightParameterValue));

        return _currentValue;
    }

    public override void BackpropagateNext(Matrix<float> previouseJacobian)
    {
        IReadOnlyVector<float> leftParameterValue = _leftParameter.CurrentValue;
        IReadOnlyVector<float> rightParameterValue = _rightParameter.CurrentValue;

        IReadOnlyMatrix<float> leftJacobian = LeftJacobian(leftParameterValue, rightParameterValue);
        IReadOnlyMatrix<float> rightJacobian = RightJacobian(leftParameterValue, rightParameterValue);
        if(leftJacobian.Size != (Dimension, LeftParameterDimension)
            || rightJacobian.Size != (Dimension, RigthParameterDimension))
            throw new InvalidOperationException();

        Matrix<float> previouseJacobianCopy = Matrix<float>.CreateCopy(previouseJacobian);
        Matrix<float> nextLeftJacobian = leftJacobian.MultiplyRightCached(previouseJacobian);
        Matrix<float> nextRightJacobian = rightJacobian.MultiplyRightCached(previouseJacobianCopy);

        _leftParameter.BackpropagateNext(nextLeftJacobian);
        _rightParameter.BackpropagateNext(nextRightJacobian);
    }
}