using System.Globalization;
using System.Numerics;
using System.Text;

namespace LinearAlgebra.Matrix;

public class Matrix<T> : IReadOnlyMatrix<T>
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

    public static Matrix<T> CreateCopy(IReadOnlyMatrix<T> other)
    {
        (int height, int width) = other.Size;
        var data = new T[height * width];

        for(int i = 0; i < height; i++)
        {
            for(int j = 0; j < width; j++)
                data[i * width + j] = other[i, j];
        }

        return new Matrix<T>(height, width, data);
    }

    public static Matrix<T> CreateCachedCopy(Matrix<T> other) 
        => new Matrix<T>(other.Height, other.Width, other._data);

    public override string ToString()
    {
        var sb = new StringBuilder();

        sb.AppendLine("[");
        for (int i = 0; i < Height; i++)
        {
            sb.Append("\t[");
            for(int j =0; j < Width; j++)
                sb.Append($"{this[i, j]},\t");
            sb.AppendLine("]");
        }
        sb.Append(']');

        return sb.ToString();
    }

    public Vector<T> ToVectorCached()
        => new Vector<T>(_data);

    public Matrix<T> MultiplyRight(IReadOnlyMatrix<T> other)
    {
        if(_width != other.Height)
            throw new ArgumentException();

        int length = Width;
           
        int height = _height;
        int width = other.Width;
        var data = new T[height * width];

        for(int i = 0; i < height; i++)
        {
            for(int j = 0; j < width; j++)
            {
                T scalarProduct = T.Zero;
                for(int k = 0; k < length; k++)
                    scalarProduct += this[i, k] * other[k, j];
                data[i * width + j] = scalarProduct;
            }
        }

        return new Matrix<T>(height, width, data);
    }

    public Matrix<T> MultiplyRightCached(Matrix<T> other)
    {
        if(_width != other.Height)
            throw new ArgumentException();

        if(IsSquareSize == false)
            return MultiplyRight(other);
           
        (int height, int width) = other.Size;
        var buffer = new T[other.Width];

        for(int i = 0; i < height; i++)
        {
            for(int j = 0; j < width; j++)
            {
                T scalarProduct = T.Zero;
                for(int k = 0; k < width; k++)
                    scalarProduct += this[i, k] * other[k, j];
                buffer[j] = scalarProduct;
            }

            for(int j = 0; j < width; j++)
                other[i, j] = buffer[j];
        }

        return CreateCachedCopy(other);
    }

}