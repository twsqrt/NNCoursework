using LinearAlgebra;

namespace NeuralNetworks.ComputationGraph;

public class VectorToMatrixNode : ReshapeNode<Vector, Matrix>
{
    private readonly int _height;
    private readonly int _width; 

    public VectorToMatrixNode(Node<Vector> input, int height, int width) 
    : base(input, new TensorShape(height, width))
    {
        if(input.Shape.Dimension != height * width)
            throw new ArgumentException();
        
        _height = height;
        _width = width;
    }

    public override Matrix CalculateValue()
        => _input.CalculateValue().AsMatrix(_height, _width);
}
