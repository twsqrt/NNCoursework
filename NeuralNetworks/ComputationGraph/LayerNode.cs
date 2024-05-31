using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class LayerNode : Node
{
    private readonly Node _child;
    private readonly ParameterNode _weights;
    private readonly Matrix _weightsMatrix;
    private readonly Matrix _weightsCachedJacobian;
    private readonly Matrix _childCachedJacobian;
    private readonly bool _shouldBackpropagateChild;
    private readonly Vector _cachedResult;
    private Vector _childValue;

    public ParameterNode Weights => _weights;
    public Node Child => _child;
    public bool ShouldBackpropagateChild => _shouldBackpropagateChild;

    public LayerNode(ParameterNode weights, Node child, int graphRootDimension, bool shouldBackpropagateChild = true) 
    : base(weights.Dimension / child.Dimension)
    {
        if(weights.Dimension % child.Dimension != 0)
            throw new ArgumentException();
        
        _child = child;
        _childValue = null;

        _weights = weights;
        _weightsMatrix = _weights.Value.AsMatrix(Dimension, _child.Dimension);
        _weightsCachedJacobian = Matrix.CreateZero(graphRootDimension, weights.Dimension);

        _cachedResult = Vector.CreateZero(Dimension);

        _shouldBackpropagateChild = shouldBackpropagateChild;

        if(shouldBackpropagateChild)
            _childCachedJacobian = Matrix.CreateZero(graphRootDimension, child.Dimension);
    }

    public override void Accept(INodeVisitor visitor)
        => visitor.Visit(this);

    public override void BackpropagateNext(Matrix previouseJacobian)
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
            Matrix.Multiply(previouseJacobian, _weightsMatrix, _childCachedJacobian);
            _child.BackpropagateNext(_childCachedJacobian);
        }
    }

    public override Vector CalculateValue()
    {
        _childValue = _child.CalculateValue();
        Matrix.Multiply(_weightsMatrix, _childValue, _cachedResult);
        return _cachedResult;
    }
}