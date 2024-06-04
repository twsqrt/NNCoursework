using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class MaxPool2DNode : Node<Tensor3D>
{
    private readonly Node<Tensor3D> _child;
    private readonly int _kernelHeight; 
    private readonly int _kernelWidth;

    public override NodeType Type => NodeType.MAXPOOL2D;

    public MaxPool2DNode(Node<Tensor3D> child, int kernelHeight, int kernelWidth)
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
    }

    public void CalculateGradientForSlice(Tensor3DSlice gradientSlice, int depth, Tensor3DSlice result)
    {
        for(int i = 0; i < gradientSlice.Height; i++)
        for(int j = 0; j < gradientSlice.Width; j++)
        {
            for(int k = 0; k < _kernelHeight; k++)
            for(int l = 0; l < _kernelWidth; l++)
            {
                int m = i * _kernelHeight + k;
                int n = j * _kernelWidth + l;

                if(_child.Value[m, n, depth] < _value[i, j, depth])
                    result[m, n] = 0.0f;
                else
                    result[m, n] = gradientSlice[i, j];
            }
        }
    }

    public override void CalculateGradient()
    {
        for(int i = 0; i < Gradient.Shape.Depth; i++)
            CalculateGradientForSlice(Gradient.Slice(i), i, _child.Gradient.Slice(i));
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

    protected override void WriteData(BinaryWriter writer)
    {
        writer.Write(_kernelHeight);
        writer.Write(_kernelHeight);
        _child.Export(writer);
    }
}
