using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class VectorToMatrixNode : Node<Matrix>
{
    private readonly Node<Vector> _child;

    public VectorToMatrixNode(Node<Vector> child, int height, int width ) 
    : base(new TensorShape3D(height, width), new INode[]{child})
    {
        _child = child;
        ParentGradient = Matrix.CreateZero(height, width);
    }

    public override void CalculateGradient()
        => _child.ParentGradient = new Vector(ParentGradient.Data);

    public override void CalculateValue()
        => _value = new Matrix(Shape.Height, Shape.Width, _child.Value.Data);
}
