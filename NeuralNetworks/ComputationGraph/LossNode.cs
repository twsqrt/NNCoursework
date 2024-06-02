using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class LossNode : Node<Vector>
{
    private readonly Node<Vector> _output;
    private readonly Node<Vector> _markup;

    public LossNode(Node<Vector> output, Node<Vector> markup) 
    : base(new TensorShape3D(1), new INode[]{output})
    {
        if(output.Shape != markup.Shape)
            throw new ArgumentException();

        _output = output;
        _markup = markup;
    }

    public override void CalculateGradient()
        => _output.ParentGradient.Scale(2.0f);

    public override void CalculateValue()
    {
        Vector.Difference(_output.Value, _markup.Value, _output.ParentGradient);
        _value = Vector.Create1D(_output.ParentGradient.LengthSquared);
    }
}
