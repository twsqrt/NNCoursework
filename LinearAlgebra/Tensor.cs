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

    public Tensor(float[] data, int height, int width = 1, int depth = 1)
    {
        if(height * width * depth != data.Length)
            throw new ArgumentException();
        
        _data = data;
        _shape = new TensorShape(height, width, depth);
    }
    
    public static Tensor Create1DShape(float[] data)
        => new Tensor(data, data.Length);

    public static Tensor Create2DShape(float[] data, int height, int width)
        => new Tensor(data, height, width);

    public static Tensor CreateZero(int height, int width = 1, int depth = 1)
    {
        var data = new float[height * width * depth];
        Array.Fill(data, 0.0f);
        return new Tensor(data, height, width, depth);
    }

    public static Tensor CreateZero(TensorShape shape)
        => CreateZero(shape.Height, shape.Width, shape.Depth);

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
