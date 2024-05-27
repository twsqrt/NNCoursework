using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class LayerNode : Node
{
    private readonly Node _child;
    private readonly ParameterNode _weights;
    private readonly Matrix<float> _weightsMatrix;
    private readonly Matrix<float> _weightsCachedJacobian;
    private readonly Matrix<float> _childCachedJacobian;
    private readonly bool _shouldBackpropagateChild;

    private Vector<float> _childValue;

    public LayerNode(ParameterNode weights, Node child, int graphRootDimension, bool shouldBackpropagateChild = true) 
    : base(weights.Dimension / child.Dimension)
    {
        if(weights.Dimension % child.Dimension != 0)
            throw new ArgumentException();
        
        _child = child;
        _weights = weights;

        _weightsMatrix = _weights.Value.AsMatrix(Dimension, _child.Dimension);
        _weightsCachedJacobian = Matrix<float>.CreateZeroMatrix(graphRootDimension, weights.Dimension);

        _shouldBackpropagateChild = shouldBackpropagateChild;
        if(_shouldBackpropagateChild)
            _childCachedJacobian = Matrix<float>.CreateZeroMatrix(graphRootDimension, child.Dimension);
        else
            _childCachedJacobian = null;


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

        if(_shouldBackpropagateChild)
        {
            Matrix<float>.Multiply(previouseJacobian, _weightsMatrix, _childCachedJacobian);
            _child.BackpropagateNext(_childCachedJacobian);
        }
    }

    public override Vector<float> CalculateValue()
    {
        _childValue = _child.CalculateValue();
        return _weightsMatrix.ApplyTo(_childValue);
    }
}