using System.Globalization;
using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class LayerNode : Node<Vector>
{
    private readonly Node<Vector> _child;
    private readonly Node<Matrix> _weights;
    private readonly Vector _weightsCachedGradient;
    private readonly Vector _childCachedGradient;
    private readonly Vector _cachedResult;
    private readonly bool _shouldBackpropagateChild;

    private Vector _childValue;
    private Matrix _weightsValue;

    public Node<Vector> Child => _child;
    public Node<Matrix> Weights => _weights;

    public bool ShouldBackpropagateChild => _shouldBackpropagateChild;

    public LayerNode(Node<Matrix> weights, Node<Vector> child, bool shouldBackpropagateChild = true) 
    : base(new TensorShape(weights.Shape.Height))
    {
        if(weights.Shape.Width != child.Shape.Height)
            throw new ArgumentException();
        
        _child = child;
        _weights = weights;

        _weightsCachedGradient = Vector.CreateZero(weights.Shape.Width * weights.Shape.Height);
        _cachedResult = Vector.CreateZero(Shape.Dimension);

        _shouldBackpropagateChild = shouldBackpropagateChild;

        if(shouldBackpropagateChild)
            _childCachedGradient = Vector.CreateZero(child.Shape.Dimension);
    }

    public override void Accept(INodeVisitor visitor)
        => visitor.Visit(this);

    public override void BackpropagateNext(Vector gradient)
    {
        int weightsWidth = _weightsValue.Width;
        for(int i = 0; i < _weightsValue.Height; i++)
        {
            for(int j = 0; j < weightsWidth; j++)
                _weightsCachedGradient[i * weightsWidth + j] = gradient[i] * _childValue[j];
        }

        _weights.BackpropagateNext(_weightsCachedGradient);

        if(_shouldBackpropagateChild)
        {
            Matrix.Multiply(gradient, _weightsValue, _childCachedGradient);
            _child.BackpropagateNext(_childCachedGradient);
        }
    }

    public override Vector CalculateValue()
    {
        _childValue = _child.CalculateValue();
        _weightsValue = _weights.CalculateValue();

        Matrix.Multiply(_weightsValue, _childValue, _cachedResult);
        return _cachedResult;
    }
}