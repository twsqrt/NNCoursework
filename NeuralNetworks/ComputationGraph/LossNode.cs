using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class LossNode : Node<float>
{
    private readonly Node<Vector> _output;
    private readonly Node<Vector> _markup;
    private Vector _difference;

    public LossNode(Node<Vector> output, Node<Vector> markup) 
    : base(new TensorShape(1))
    {
        if(output.Shape != markup.Shape)
            throw new ArgumentException();

        _output = output;
        _markup = markup;

        _difference = Vector.CreateZero(output.Shape.Height);
    }

    public override void Accept(INodeVisitor visitor)
        => throw new InvalidOperationException();

    public override void BackpropagateNext(float gradient)
    {
        _difference.Scale(2.0f * gradient);
        _output.BackpropagateNext(_difference);
    }

    public override float CalculateValue()
    {
        Vector.Difference(_output.CalculateValue(), _markup.CalculateValue(), _difference);
        return _difference.LengthSquared;
    }
}
