using System.Numerics;

namespace LinearAlgebra;

public class IdentityMatrix<T> : IReadOnlyMatrix<T>
    where T : INumber<T>
{
    private readonly int _size;

    public T this[int i, int j] 
        => (i == j) ? T.One : T.Zero;

    public (int, int) Size => (_size, _size);

    public bool IsSquareSize => true;

    public int Width => _size;

    public int Height => _size;

    public IdentityMatrix(int size)
    {
        _size = size;
    }

    public Matrix<T> MultiplyRight(IReadOnlyMatrix<T> other)
        => Matrix<T>.CreateCopy(other);

    public Matrix<T> MultiplyRightCached(Matrix<T> other)
        => Matrix<T>.CreateCachedCopy(other);

    public Vector<T> ApplyTo(Vector<T> vector)
        => vector;
}
