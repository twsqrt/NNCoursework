using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class Convolution2DNode : Node<Tensor3D>
{
    private readonly Node<Matrix> _child;
    private readonly Node<Tensor3D> _kernel; 
    private readonly bool _shouldBackpropagateChild;

    public bool ShouldBackpropagateChild => _shouldBackpropagateChild;

    public Convolution2DNode(Node<Matrix> child, Node<Tensor3D> kernel, bool shouldBackpropagateChild = true)
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
   }

    public override void CalculateGradient()
    {
        for(int i = 0; i < _kernel.Shape.Depth; i++)
        {
            Tensor3DSlice gradientSlice = Gradient.Slice(i);
            Tensor3DSlice resultSlice = _kernel.Gradient.Slice(i);
            Matrix.Convolution(_child.Value, gradientSlice, resultSlice);
        }

        if(_shouldBackpropagateChild)
        {
            //...
        }
    }

    public override void CalculateValue()
    {
        for(int i = 0; i < _kernel.Shape.Depth; i++)
        {
            Tensor3DSlice kernelSlice = _kernel.Value.Slice(i);
            Tensor3DSlice resultSlice = _value.Slice(i);
            Matrix.Convolution(_child.Value, kernelSlice, resultSlice);
        }
    }
}
