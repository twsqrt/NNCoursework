using System.Numerics;

namespace LinearAlgebra;

public interface IReadOnlyMatrix<T>
    where T : INumber<T>
{
    T this[int i, int j] { get; }
    (int, int) Size { get; }
    bool IsSquareSize { get;}
    int Width { get; }
    int Height { get; }

    Matrix<T> MultiplyRight(IReadOnlyMatrix<T> other);
    Matrix<T> MultiplyRightCached(Matrix<T> other);

    Vector<T> ApplyTo(IReadOnlyVector<T> vector);
}