using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class MaxPool2DNode : Node<Tensor>
{
    private readonly Node<Tensor> _child;
    private readonly int _kernelHeight; 
    private readonly int _kernelWidth;
    private readonly Tensor _childCachedGradient;

    public MaxPool2DNode(Node<Tensor> child, int kernelHeight, int kernelWidth)
    : base(new TensorShape(
        child.Shape.Height / kernelHeight, 
        child.Shape.Width / kernelWidth, 
        child.Shape.Depth), new INode[]{child})
    {
        if(child.Shape.Height % kernelHeight != 0
            || child.Shape.Width % kernelWidth != 0)
            throw new ArgumentException();
        
        _child = child;
        _kernelHeight = kernelHeight;
        _kernelWidth = kernelWidth;

        _value = Tensor.CreateZero(Shape);
        _childCachedGradient = Tensor.CreateZero(child.Shape);
    }

    public void CalculateGradientForSlice(Matrix gradientSlice, int depth, Matrix result)
    {
        for(int i = 0; i < gradientSlice.Height; i++)
        for(int j = 0; j < gradientSlice.Width; j++)
        {
            for(int k = 0; k < _kernelHeight; k++)
            for(int l = 0; l < _kernelWidth; l++)
            {
                if(_child.Value[i + k, j + l, depth] < _value[i, j, depth])
                    result[i + k, j + l] = 0.0f;
                else
                    result[i + k, j + l] = gradientSlice[i, j];
            }
        }
    }

    public override void BackpropagateNext(Tensor gradient)
    {
        for(int i = 0; i < gradient.Shape.Depth; i++)
            CalculateGradientForSlice(gradient.Slice(i), i, _childCachedGradient.Slice(i));
        
        _child.BackpropagateNext(_childCachedGradient);
    }

    public override void CalculateValue()
    {
        for(int i = 0; i < _child.Value.Shape.Depth; i++)
        {
            Matrix childResultSlice = _child.Value.Slice(i);
            Matrix resultSlice = _value.Slice(i);
            Matrix.MaxPool(childResultSlice, _kernelHeight, _kernelWidth, resultSlice);
        }
    }
}
