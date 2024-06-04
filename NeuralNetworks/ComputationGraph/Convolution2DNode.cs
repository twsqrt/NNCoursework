using LinearAlgebra;
using NeuralNetworks.ComputationGraph;

namespace NeuralNetworks;

public class Convolution2DNode : Node<Tensor3D>
{
    private readonly Node<Matrix> _child;
    private readonly Node<Tensor3D> _kernel; 
    private readonly bool _shouldBackpropagateChild;

    public bool ShouldBackpropagateChild => _shouldBackpropagateChild;

    public override NodeType Type => NodeType.CONVOLUTION2D;

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
            for(int i = 0; i < _child.Shape.Height; i++)
            for(int j = 0; j < _child.Shape.Width; j++)
            {
                int kMin = Math.Max(i - _kernel.Shape.Height + 1, 0);
                int kMax = Math.Min(_child.Shape.Height - _kernel.Shape.Height, i);

                int lMin = Math.Max(j - _kernel.Shape.Width + 1, 0);
                int lMax = Math.Min(_child.Shape.Width - _kernel.Shape.Width, j);

                float sum = 0.0f;

                for(int depth = 0; depth < _kernel.Shape.Depth; depth++)
                for(int k = kMin; k <= kMax; k++)
                for(int l = lMin; l <= lMax; l++)
                    sum += Gradient[k, l, depth] * _kernel.Value[i - k, j - l, depth];

                _child.Gradient[i, j] = sum;
            }
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

    protected override void WriteData(BinaryWriter writer)
    {
        writer.Write(_shouldBackpropagateChild);
        _child.Export(writer);
        _kernel.Export(writer);
    }
}
