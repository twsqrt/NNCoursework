using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class SoftMaxNode : Node<Vector>
{
    private readonly Node<Vector> _child;
    private readonly Vector _childExponents;
    private float _childExponentsSum;

    public SoftMaxNode(Node<Vector> child) 
    : base(child.Shape, new INode[]{child})
    {
        _child = child;

        _childExponents = Vector.CreateZero(Shape.Dimension);
    }

    public override NodeType Type => NodeType.SOFTMAX;

    public override void CalculateGradient()
    {
        float t = 0.0f;
        for(int i = 0; i < Shape.Dimension; i++)
            t += Gradient[i] * _value[i];

        for(int i = 0; i < Shape.Dimension; i++)
            _child.Gradient[i] = _value[i] * (Gradient[i] - t);
    }

    public override void CalculateValue()
    {
        float maxElement = _child.Value[0];

        for(int i = 1; i < Shape.Dimension; i++)
        {
            if(maxElement < _child.Value[i])
                maxElement = _child.Value[i];
        }

        _childExponentsSum = 0.0f;

        for(int i = 0; i < Shape.Dimension; i++)
        {
            _childExponents[i] = MathF.Exp(_child.Value[i] - maxElement);
            _childExponentsSum += _childExponents[i];
        }
        
        for(int i = 0; i < _value.Dimension; i++)
            _value[i] = _childExponents[i] / _childExponentsSum;
    }

    protected override void WriteData(BinaryWriter writer)
        => _child.Export(writer);
}
