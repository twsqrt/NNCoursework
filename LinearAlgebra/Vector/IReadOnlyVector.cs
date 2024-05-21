using System.Numerics;

namespace LinearAlgebra;

public interface IReadOnlyVector<T>
    where T : INumber<T>
{
    T this[int index] { get; }
    int Dimension { get; }

    T LengthSquared { get; }

    T ToNumber();

    IReadOnlyMatrix<T> ToMatrixCached(int height, int width);
}
