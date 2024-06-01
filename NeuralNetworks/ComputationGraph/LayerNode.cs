using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class LayerNode : Node<Vector>
{
    private readonly Node<Vector> _child;
    private readonly Node<Matrix> _weights;
    private readonly Matrix _weightsCachedGradient;
    private readonly Vector _childCachedGradient;
    private readonly bool _shouldBackpropagateChild;

    public LayerNode(Node<Matrix> weights, Node<Vector> child, bool shouldBackpropagateChild = true) 
    : base(new TensorShape(weights.Shape.Height), new INode[]{weights, child})
    {
        if(weights.Shape.Width != child.Shape.Height)
            throw new ArgumentException();
        
        _child = child;
        _weights = weights;
        _shouldBackpropagateChild = shouldBackpropagateChild;

        _weightsCachedGradient = Matrix.CreateZero(weights.Shape.Height, weights.Shape.Width);
        _value = Vector.CreateZero(Shape.Dimension);

        if(shouldBackpropagateChild)
            _childCachedGradient = Vector.CreateZero(child.Shape.Dimension);
    }

    public override void BackpropagateNext(Vector gradient)
    {
        for(int i = 0; i < _weightsCachedGradient.Height; i++)
        for(int j = 0; j < _weightsCachedGradient.Width; j++)
                _weightsCachedGradient[i, j] = gradient[i] * _child.Value[j];

        _weights.BackpropagateNext(_weightsCachedGradient);

        if(_shouldBackpropagateChild)
        {
            Matrix.Multiply(gradient, _weights.Value, _childCachedGradient);
            _child.BackpropagateNext(_childCachedGradient);
        }
    }

    public override void CalculateValue()
        => Matrix.Multiply(_weights.Value, _child.Value, _value);
}