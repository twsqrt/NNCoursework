using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class LossNode : Node<float>
{
    private readonly Node<Vector> _output;
    private readonly Node<Vector> _markup;
    private readonly Vector _difference;

    public LossNode(Node<Vector> output, Node<Vector> markup) 
    : base(new TensorShape(1), new INode[]{output})
    {
        if(output.Shape != markup.Shape)
            throw new ArgumentException();

        _output = output;
        _markup = markup;

        _difference = Vector.CreateZero(output.Shape.Dimension);
    }

    public override void BackpropagateNext(float gradient)
    {
        _difference.Scale(2.0f * gradient);
        _output.BackpropagateNext(_difference);
    }

    public override void CalculateValue()
    {
        Vector.Difference(_output.Value, _markup.Value, _difference);
        _value = _difference.LengthSquared;
    }
}
