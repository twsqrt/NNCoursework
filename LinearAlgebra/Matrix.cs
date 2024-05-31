namespace LinearAlgebra;

public class Matrix
{
    private readonly int _height;
    private readonly int _width;
    private readonly float[] _data;

    public int Height => _height;
    public int Width => _width;
    public (int, int) Shape => (_height, _width);

    public float this[int i, int j] 
    {
        get => _data[i * _width + j];
        set => _data[i * _width + j] = value;
    }

    public Matrix(int height, int width, float[] data)
    {
        if(data.Length != height * width)
            throw new ArgumentException();

        _height = height;
        _width = width;
        _data = data;
    }

    public static Matrix CreateZero(int height, int width)
    {
        var data = new float[height * width];
        Array.Fill(data, 0.0f);
        return new Matrix(height, width, data);
    }

    public static Matrix CreateIdentity(int size)
    {
        var data = new float[size * size];
        for(int i = 0; i < size; i++)
        {
            for(int j = 0; j < size; j++)
                data[i * size + j] = i == j ? 1.0f : 0.0f;
        }

        return new Matrix(size, size, data);
    }

    public Vector AsVector()
        => new Vector(_data);

    public void CopyValuesFrom(Matrix other)
    {
        for(int i = 0; i < _data.Length; i++)
            _data[i] = other._data[i];
    }

    public void CopyValuesFrom(Vector other)
    {
        for(int i = 0; i < _data.Length; i++)
            _data[i] = other[i];
    }

    public static void Multiply(Matrix lhs, Matrix rhs, Matrix result)
    {
        for(int i = 0; i < lhs.Height; i++)
        {
            for(int j = 0; j < rhs.Width; j++)
            {
                float sum = 0.0f;
                for(int k = 0; k < lhs.Width; k++)
                    sum += lhs[i, k] * rhs[k, j];
                
                result[i, j] = sum;
            }
        }
    }

    public static void Multiply(Matrix lhs, Vector rhs, Vector result)
    {
        for(int i = 0; i < lhs.Height; i++)
        {
            float sum = 0.0f;
            for(int j = 0; j < rhs.Dimension;  j++)
                sum += rhs[j] * lhs[i, j];

            result[i] = sum;
        }
    }

    public static void Multiply(Vector lhs, Matrix rhs, Vector result)
    {
        for(int i = 0; i < rhs.Height; i++)
        {
            float sum = 0.0f;
            for(int j = 0; j < lhs.Dimension;  j++)
                sum += lhs[j] * rhs[j, i];

            result[i] = sum;
        }
    }
}