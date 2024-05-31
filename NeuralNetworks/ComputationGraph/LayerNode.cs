using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class LayerNode : Node
{
    private readonly Node _child;
    private readonly ParameterNode _weights;
    private readonly Matrix _weightsMatrix;
    private readonly Vector _weightsCachedGradient;
    private readonly Vector _childCachedGradient;
    private readonly Vector _cachedResult;
    private readonly bool _shouldBackpropagateChild;
    private Vector _childValue;

    public ParameterNode Weights => _weights;
    public Node Child => _child;
    public bool ShouldBackpropagateChild => _shouldBackpropagateChild;

    public LayerNode(ParameterNode weights, Node child, bool shouldBackpropagateChild = true) 
    : base(weights.Dimension / child.Dimension)
    {
        if(weights.Dimension % child.Dimension != 0)
            throw new ArgumentException();
        
        _child = child;
        _childValue = null;

        _weights = weights;
        _weightsMatrix = _weights.Value.AsMatrix(Dimension, _child.Dimension);
        _weightsCachedGradient = Vector.CreateZero(weights.Dimension);

        _cachedResult = Vector.CreateZero(Dimension);

        _shouldBackpropagateChild = shouldBackpropagateChild;

        if(shouldBackpropagateChild)
            _childCachedGradient = Vector.CreateZero(child.Dimension);
    }

    public override void Accept(INodeVisitor visitor)
        => visitor.Visit(this);

    public override void BackpropagateNext(Vector gradient)
    {
        int weightsWidth = _weightsMatrix.Width;
        for(int i = 0; i < _weightsMatrix.Height; i++)
        {
            for(int j = 0; j < weightsWidth; j++)
                _weightsCachedGradient[i * weightsWidth + j] = gradient[i] * _childValue[j];
        }

        _weights.BackpropagateNext(_weightsCachedGradient);

        if(_shouldBackpropagateChild)
        {
            Matrix.Multiply(gradient, _weightsMatrix, _childCachedGradient);
            _child.BackpropagateNext(_childCachedGradient);
        }
    }

    public override Vector CalculateValue()
    {
        _childValue = _child.CalculateValue();
        Matrix.Multiply(_weightsMatrix, _childValue, _cachedResult);
        return _cachedResult;
    }
}