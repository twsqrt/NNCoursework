using System.Numerics;
using System.Text;

namespace LinearAlgebra;

public class Vector<T>
    where T : INumber<T>
{
    private readonly int _length;
    private readonly T[] _vector;

    public int Length => _length;

    public T this[int index]
    {
        get => _vector[index];
        set => _vector[index] = value;
    }

    private Vector(int length)
    {
        _length = length;
        _vector = new T[length];
    }

    public Vector(T[] vector)
    {
        _length = vector.Length;
        _vector = vector;
    }

    public static Vector<T> ZeroVector(int length)
        => new Vector<T>(Enumerable.Range(0, length).Select(_ => T.Zero).ToArray());
    
    public static Vector<T> CreateFromFunction(int length, Func<int, T> function)
    {
        var result = new Vector<T>(length);
        for(int i =0; i < length; i++)
            result[i] = function(i);
          
        return result;
    }
    
    public static T ScalarProduct(Vector<T> lhs, Vector<T> rhs)
    {
        if(lhs.Length != rhs.Length)
            throw new ArgumentException();
        
        T result = T.Zero;
        for(int i =0; i < lhs.Length; i++)
            result += lhs[i] * rhs[i];
        return result;
    }

    public override string ToString()
    {
        var sb = new StringBuilder();

        sb.Append("[");
        for(int i =0; i < _length; i++)
            sb.Append($"{_vector[i]},\t");
        sb.Append("]");

        return sb.ToString();
    }

    public static Vector<T> operator *(T scalar, Vector<T> vector)
        => CreateFromFunction(vector.Length, i => scalar * vector[i]);

    public static Vector<T> operator +(Vector<T> vector) => vector;
    public static Vector<T> operator -(Vector<T> vector) => - T.One * vector;

    public static Vector<T> operator +(Vector<T> lhs, Vector<T> rhs)
    {
        if(lhs.Length != rhs.Length)
            throw new ArgumentException();
        
        return CreateFromFunction(lhs.Length, i => lhs[i] + rhs[i]);
    }

    public static Vector<T> operator -(Vector<T> lhs, Vector<T> rhs)
    {
        if(lhs.Length != rhs.Length)
            throw new ArgumentException();
        
        return CreateFromFunction(lhs.Length, i => lhs[i] - rhs[i]);
    }

    public static T operator *(Vector<T> lhs, Vector<T> rhs)
        => ScalarProduct(lhs, rhs);
}
