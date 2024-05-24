using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class LayerNode : Node
{
    private readonly Node _child;
    private readonly Node _weights;
    private readonly Matrix<float> _weightsMatrix;
    private readonly Matrix<float> _weightsCachedJacobian;
    private Vector<float> _childValue;

    public LayerNode(Node weights, Node child, int graphRootDimension) : base(weights.Dimension / child.Dimension)
    {
        if(weights.Dimension % child.Dimension != 0)
            throw new ArgumentException();
        
        _child = child;
        _weights = weights;
        _weightsMatrix = Matrix<float>.CreateZeroMatrix(Dimension, child.Dimension);
        _weightsCachedJacobian = Matrix<float>.CreateZeroMatrix(graphRootDimension, weights.Dimension);

        _childValue = null;
    }

    public override void BackpropagateNext(Matrix<float> previouseJacobian)
    {
        int weightsWidth = _weightsMatrix.Width;
        for(int i = 0; i < _weightsCachedJacobian.Height; i++)
        {
            for(int j = 0; j < _weightsMatrix.Height; j++)
            {
                for(int k = 0; k < weightsWidth; k++)
                    _weightsCachedJacobian[i, j * weightsWidth + k] = previouseJacobian[i, j] * _childValue[k];
            }
        }

        _weights.BackpropagateNext(_weightsCachedJacobian);

        Matrix<float> childJacobian = _weightsMatrix.MultiplyRightCached(previouseJacobian);
        _child.BackpropagateNext(childJacobian);
    }

    public override Vector<float> CalculateValue()
    {
        Vector<float> weightsValues = _weights.CalculateValue();
        _weightsMatrix.CopyValuesFrom(weightsValues);
        _childValue = _child.CalculateValue();

        return _weightsMatrix.ApplyTo(_childValue);
    }
}