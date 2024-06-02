using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class MaxPool2DNode : Node<Tensor3D>
{
    private readonly Node<Tensor3D> _child;
    private readonly int _kernelHeight; 
    private readonly int _kernelWidth;

    public MaxPool2DNode(Node<Tensor3D> child, int kernelHeight, int kernelWidth)
    : base(new TensorShape3D(
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

        _value = Tensor3D.CreateZero(Shape);
        ParentGradient = Tensor3D.CreateZero(Shape);
    }

    public void CalculateGradientForSlice(Tensor3DSlice gradientSlice, int depth, Tensor3DSlice result)
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

    public override void CalculateGradient()
    {
        for(int i = 0; i < ParentGradient.Shape.Depth; i++)
            CalculateGradientForSlice(ParentGradient.Slice(i), i, _child.ParentGradient.Slice(i));
    }

    public override void CalculateValue()
    {
        for(int i = 0; i < _child.Value.Shape.Depth; i++)
        {
            Tensor3DSlice childResultSlice = _child.Value.Slice(i);
            Tensor3DSlice resultSlice = _value.Slice(i);
            Matrix.MaxPool(childResultSlice, _kernelHeight, _kernelWidth, resultSlice);
        }
    }
}
