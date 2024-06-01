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

    public override void BackpropagateNext(Matrix gradient)
        => _input.BackpropagateNext(gradient.AsVector());

    public override void CalculateValue()
    {
        _value = _input.Value.AsMatrix(_height, _width);
    }
}
