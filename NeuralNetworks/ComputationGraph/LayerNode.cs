using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class LayerNode : Node<Vector>
{
    private readonly Node<Vector> _child;
    private readonly Node<Matrix> _weights;
    private readonly bool _shouldBackpropagateChild;

    public LayerNode(Node<Matrix> weights, Node<Vector> child, bool shouldBackpropagateChild = true) 
    : base(new TensorShape(weights.Shape.Height), new INode[]{weights, child})
    {
        if(weights.Shape.Width != child.Shape.Height)
            throw new ArgumentException();
        
        _child = child;
        _weights = weights;
        _shouldBackpropagateChild = shouldBackpropagateChild;
    }

    public override NodeType Type => NodeType.LAYER;

    public override void CalculateGradient()
    {
        for(int i = 0; i < _weights.Gradient.Height; i++)
        for(int j = 0; j < _weights.Gradient.Width; j++)
                _weights.Gradient[i, j] = Gradient[i] * _child.Value[j];

        if(_shouldBackpropagateChild)
            Matrix.Multiply(Gradient, _weights.Value, _child.Gradient);
    }

    public override void CalculateValue()
        => Matrix.Multiply(_weights.Value, _child.Value, _value);

    protected override void WriteData(BinaryWriter writer)
    {
        writer.Write(_shouldBackpropagateChild);
        _child.Export(writer);
        _weights.Export(writer);
    }
}