using System.Numerics;

namespace LinearAlgebra;

public interface IReadOnlyVector<T>
    where T : INumber<T>
{
    T this[int index] { get; }
    int Length { get; }

    T ToNumber();
}
