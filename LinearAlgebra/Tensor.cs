using System.Runtime.InteropServices;

namespace LinearAlgebra;

public class Tensor
{
    private readonly float[] _data;
    private readonly TensorShape _shape;

    public TensorShape Shape => _shape;

    public float this[int i] 
    {
        get => _data[i];
        set => _data[i] = value;
    }

    public float this[int i, int j, int k]
    {
        get => _data[i * _shape.Width + j + k * _shape.Height * _shape.Width];
        set => _data[i * _shape.Width + j + k * _shape.Height * _shape.Width] = value;
    }

    public Matrix Slice(int depth)
    {
        int dataStartIndex = _shape.Height * _shape.Width * depth;
        return new Matrix(Shape.Height, Shape.Width, _data, dataStartIndex);
    }

    public Tensor(float[] data, TensorShape shape)
    {
        if(shape.Dimension != data.Length)
            throw new ArgumentException();
        
        _data = data;
        _shape = shape;
    }
    
    public static Tensor CreateZero(TensorShape shape)
    {
        var data = new float[shape.Dimension];
        Array.Fill(data, 0.0f);
        return new Tensor(data, shape);
    }
     
    public Vector AsVector()
        => new Vector(_data);

    public void Scale(float scalar)
    {
        for(int i = 0; i < _data.Length; i++)
            _data[i] *= scalar;
    }

    public static void Addition(Tensor lhs, Tensor rhs, Tensor result)
    {
        for(int i = 0; i < result._data.Length; i++)
            result._data[i] = lhs._data[i] + rhs._data[i];
    }

    public static void Difference(Tensor lhs, Tensor rhs, Tensor result)
    {
        for(int i = 0; i < result._data.Length; i++)
            result._data[i] = lhs._data[i] - rhs._data[i];
    }
}
