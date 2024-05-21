using System.Numerics;

namespace LinearAlgebra;

public class DiagonalMatrix<T> : IReadOnlyMatrix<T>
    where T : INumber<T>
{
    private readonly int _length;
    private readonly T[] _diagonalElements;

    public T this[int i, int j]
        => (i == j) ? _diagonalElements[i] : T.Zero;

    public (int, int) Size => (_length, _length);

    public int Width => _length;

    public int Height => _length;

    public bool IsSquareSize => true;

    public DiagonalMatrix(T[] diagonalElements)
    {
        _length = diagonalElements.Length;
        _diagonalElements = diagonalElements;
    }

    public Matrix<T> MultiplyRight(IReadOnlyMatrix<T> other)
    {
        if(_length != other.Width)
            throw new ArgumentException();

        (int height, int width) = other.Size;
        var data = new T[height * width];

        for(int i = 0; i < height; i++)
        {
            for(int j = 0; j < width; j++)
                data[i * width + j] = other[i, j] * _diagonalElements[i];
        }
           
        return new Matrix<T>(height, width, data);
    }

    public Matrix<T> MultiplyRightCached(Matrix<T> other)
    {
        if(_length != other.Width)
            throw new ArgumentException();
           
        (int height, int width) = other.Size;

        for(int i = 0; i < height; i++)
        {
            for(int j = 0; j < width; j++)
                other[i, j] *= _diagonalElements[j];
        }

        return Matrix<T>.CreateCachedCopy(other);
    }

    public Vector<T> ApplyTo(IReadOnlyVector<T> vector)
    {
        if(vector.Dimension != _length)
            throw new ArgumentException();

        var data = new T[vector.Dimension];
        for(int i = 0; i < vector.Dimension; i++)
            data[i] = _diagonalElements[i] * vector[i];

        return new Vector<T>(data);
    }
}
