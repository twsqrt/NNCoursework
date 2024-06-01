using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class Convolution2DNode : Node<Tensor>
{
    private readonly Node<Matrix> _child;
    private readonly Node<Tensor> _kernel; 
    private readonly Tensor _kernelCachedGradient;
    private readonly bool _shouldBackpropagateChild;

    public Convolution2DNode(Node<Matrix> child, Node<Tensor> kernel, bool shouldBackpropagateChild = true)
    : base(new TensorShape(
        child.Shape.Height - kernel.Shape.Height + 1, 
        child.Shape.Width - kernel.Shape.Width + 1, 
        kernel.Shape.Depth), new INode[]{child, kernel})
    {
        if(kernel.Shape.Height > child.Shape.Height
            || kernel.Shape.Width > child.Shape.Width)
            throw new ArgumentException(); 

        _child = child;
        _kernel = kernel;

        _shouldBackpropagateChild = shouldBackpropagateChild;

        _value = Tensor.CreateZero(Shape);
        _kernelCachedGradient = Tensor.CreateZero(kernel.Shape);
    }

    public override void BackpropagateNext(Tensor gradient)
    {
        for(int i = 0; i < _kernel.Shape.Depth; i++)
        {
            Matrix gradientSlice = gradient.Slice(i);
            Matrix resultSlice = _kernelCachedGradient.Slice(i);
            Matrix.Convolution(_child.Value, gradientSlice, resultSlice);
        }

        _kernel.BackpropagateNext(_kernelCachedGradient);

        if(_shouldBackpropagateChild)
        {
            //...
            // child.BackpropagateNext();
        }
    }

    public override void CalculateValue()
    {
        for(int i = 0; i < _kernel.Shape.Depth; i++)
        {
            Matrix kernelSlice = _kernel.Value.Slice(i);
            Matrix resultSlice = _value.Slice(i);
            Matrix.Convolution(_child.Value, kernelSlice, resultSlice);
        }
    }
}
