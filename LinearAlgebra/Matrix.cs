using System.Numerics;

namespace LinearAlgebra;

public class Matrix<T>
    where T : INumber<T>
{
    private readonly int _height;
    private readonly int _width;
    private readonly T[] _data;

    public (int, int) Size => (_height, _width);
    public int Height => _height;
    public int Width => _width;

    public bool IsSquareSize => _height == _width;

    public T this[int i, int j] 
    {
        get => _data[i * _width + j];
        set => _data[i * _width + j] = value;
    }

    public Matrix(int height, int width, T[] data)
    {
        if(data.Length != height * width)
            throw new ArgumentException();

        _height = height;
        _width = width;
        _data = data;
    }

    public static Matrix<T> CreateZeroMatrix(int height, int width)
    {
        var data = new T[height * width];
        Array.Fill<T>(data, T.Zero);

        return new Matrix<T>(height, width, data);
    }

    public static Matrix<T> CreateIdentityMatrix(int size)
    {
        var data = new T[size * size];
        for(int i = 0; i < size; i++)
        {
            for(int j = 0; j < size; j++)
                data[i * size + j] = i == j ? T.One : T.Zero;
        }

        return new Matrix<T>(size, size, data);
    }

    public Vector<T> AsVector()
        => new Vector<T>(_data);

    public void CopyValuesFrom(Matrix<T> other)
    {
        for(int i = 0; i < _data.Length; i++)
            _data[i] = other._data[i];
    }

    public void CopyValuesFrom(Vector<T> other)
    {
        for(int i = 0; i < _data.Length; i++)
            _data[i] = other[i];
    }

    public static void Multiply(Matrix<T> lhs, Matrix<T> rhs, Matrix<T> result)
    {
        for(int i = 0; i < lhs.Height; i++)
        {
            for(int j = 0; j < rhs.Width; j++)
            {
                T sum = T.Zero;
                for(int k = 0; k < lhs.Width; k++)
                    sum += lhs[i, k] * rhs[k, j];
                
                result[i, j] = sum;
            }
        }
    }

    public Vector<T> ApplyTo(Vector<T> vector)
    {
        if(vector.Dimension != _width)
            throw new ArgumentException();

        var data = new T[_height];

        for(int i = 0; i < _height; i++)
        {
            T scalarProduct = T.Zero;
            for(int j = 0; j < vector.Dimension;  j++)
                scalarProduct += vector[j] * this[i, j];

            data[i] = scalarProduct;
        }

        return new Vector<T>(data);
    }
}