using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class Convolution2DNode : Node<Tensor>
{
    private readonly Node<Tensor> _child;
    private readonly Node<Tensor> _kernel; 
    private readonly Tensor _cachedResult;
    private Tensor _childResult;

    public Convolution2DNode(Node<Tensor> child, Node<Tensor> kernel) 
    : base(new TensorShape(
        child.Shape.Height - kernel.Shape.Height + 1, 
        child.Shape.Width - kernel.Shape.Width + 1, 
        child.Shape.Depth * kernel.Shape.Depth))
    {
        if(kernel.Shape.Height > child.Shape.Height
            || kernel.Shape.Width > child.Shape.Width)
            throw new ArgumentException(); 

        _child = child;
        _kernel = kernel;

        _cachedResult = Tensor.CreateZero(Shape);
    }

    public override void Accept(INodeVisitor visitor)
    {
        throw new NotImplementedException();
    }

    public override void BackpropagateNext(Tensor gradient)
        => throw new NotImplementedException();

    public override Tensor CalculateValue()
    {
        _childResult = _child.CalculateValue();
        Tensor kernelResult = _kernel.CalculateValue();

        int numberOfChannels = kernelResult.Shape.Depth;

        for(int i = 0; i < _child.Shape.Depth; i++)
        {
            Matrix slice = _childResult.Slice(i);
            for(int j = 0; j < numberOfChannels; j++)
            {
                Matrix kernelSlice = kernelResult.Slice(j);
                Matrix resultSlice = _cachedResult.Slice(numberOfChannels * i + j);
                Matrix.Convolution(slice, kernelSlice, resultSlice);
            }
        }

        return _cachedResult;
    }
}
