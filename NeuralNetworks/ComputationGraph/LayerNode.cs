using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class LayerNode : Node<Vector>
{
    private readonly Node<Vector> _child;
    private readonly Node<Matrix> _weights;
    private readonly bool _shouldBackpropagateChild;

    public LayerNode(Node<Matrix> weights, Node<Vector> child, bool shouldBackpropagateChild = true) 
    : base(new TensorShape3D(weights.Shape.Height), new INode[]{weights, child})
    {
        if(weights.Shape.Width != child.Shape.Height)
            throw new ArgumentException();
        
        _child = child;
        _weights = weights;
        _shouldBackpropagateChild = shouldBackpropagateChild;

        _value = Vector.CreateZero(Shape.Dimension);
        ParentGradient = Vector.CreateZero(Shape.Dimension);
    }

    public override void CalculateGradient()
    {
        for(int i = 0; i < _weights.ParentGradient.Height; i++)
        for(int j = 0; j < _weights.ParentGradient.Width; j++)
                _weights.ParentGradient[i, j] = ParentGradient[i] * _child.Value[j];

        if(_shouldBackpropagateChild)
            Matrix.Multiply(ParentGradient, _weights.Value, _child.ParentGradient);
    }

    public override void CalculateValue()
        => Matrix.Multiply(_weights.Value, _child.Value, _value);
}