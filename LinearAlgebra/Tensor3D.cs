namespace LinearAlgebra;

public class Tensor3D : ITensor
{
    private readonly float[] _data;
    private readonly TensorShape3D _shape;

    public TensorShape3D Shape => _shape;
    public float[] Data => _data;

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

    public Tensor3DSlice Slice(int depth)
    {
        int startIndex = _shape.Height * _shape.Width * depth;
        return new Tensor3DSlice(Shape.Height, Shape.Width, _data, startIndex);
    }

    public Tensor3D(float[] data, TensorShape3D shape)
    {
        if(shape.Dimension != data.Length)
            throw new ArgumentException();
        
        _data = data;
        _shape = shape;
    }
    
    public void Scale(float scalar)
    {
        for(int i = 0; i < _data.Length; i++)
            _data[i] *= scalar;
    }

    public static void Addition(Tensor3D lhs, Tensor3D rhs, Tensor3D result)
    {
        for(int i = 0; i < result._data.Length; i++)
            result._data[i] = lhs._data[i] + rhs._data[i];
    }

    public static void Difference(Tensor3D lhs, Tensor3D rhs, Tensor3D result)
    {
        for(int i = 0; i < result._data.Length; i++)
            result._data[i] = lhs._data[i] - rhs._data[i];
    }
}
