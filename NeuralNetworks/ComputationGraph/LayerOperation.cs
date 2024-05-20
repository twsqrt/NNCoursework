using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class LayerOperation : Node
{
    private readonly Node _parameter;
    private readonly Node _weights;
    private readonly int _weightsHeight;
    private readonly int _weightsWidth;

    public LayerOperation(Node weights, Node parameter) : base(weights.Dimension / parameter.Dimension)
    {
        if(weights.Dimension % parameter.Dimension != 0)
            throw new ArgumentException();

        _parameter = parameter;

        _weights = weights;
        _weightsHeight = weights.Dimension / parameter.Dimension;
        _weightsWidth = parameter.Dimension;
    }

    public override void BackpropagateNext(Matrix<float> previouseJacobian)
    {
        IReadOnlyMatrix<float> weightsMatrix = _weights.CurrentValue.ToMatrixCached(_weightsWidth, _weightsWidth);
        IReadOnlyVector<float> parameterValue = _parameter.CurrentValue;

        Matrix<float> weightsJacobian = Matrix<float>.ZeroMatrix(previouseJacobian.Height, _weights.Dimension);
        for(int i = 0; i < weightsJacobian.Height; i++)
        {
            for(int j = 0; j < _weightsHeight; j++)
            {
                for(int k = 0; k < _weightsWidth; k++)
                    weightsJacobian[i, j * _weightsWidth + k] = previouseJacobian[i, j] * parameterValue[k];
            }
        }

        _weights.BackpropagateNext(weightsJacobian);

        Matrix<float> parameterJacobian = weightsMatrix.MultiplyRightCached(previouseJacobian);
        _parameter.BackpropagateNext(parameterJacobian);
    }

    public override IReadOnlyVector<float> UpdateValue()
    {
        IReadOnlyMatrix<float> weightsMatrix = _weights.UpdateValue().ToMatrixCached(_weightsHeight, _weightsWidth);
        IReadOnlyVector<float> parameterValue = _parameter.UpdateValue();
        Vector<float> result = weightsMatrix.ApplyTo(parameterValue);

        _currentValue.SetValue(result);

        return result;
    }
}