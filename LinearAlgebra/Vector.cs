using System.Numerics;
using System.Text;

namespace LinearAlgebra;

public class Vector<T> : IReadOnlyVector<T>
    where T : INumber<T>
{
    private readonly int _length;
    private readonly T[] _data;

    public int Length => _length;

    public T this[int index]
    {
        get => _data[index];
        set => _data[index] = value;
    }

    public Vector(T[] data)
    {
        _length = data.Length;
        _data = data;
    }

    public static Vector<T> Copy(IReadOnlyVector<T> other)
    {
        var data = new T[other.Length];
        for(int i = 0; i < other.Length; i++)
            data[i] = other[i];
           
        return new Vector<T>(data);
    }

    public static Vector<T> ZeroVector(int length)
        => new Vector<T>(Enumerable.Range(0, length).Select(_ => T.Zero).ToArray());
    
    public static Vector<T> CreateFromFunction(int length, Func<int, T> function)
    {
        var data = new T[length];
        for(int i =0; i < length; i++)
            data[i] = function(i);
          
        return new Vector<T>(data);
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
            sb.Append($"{_data[i]},\t");
        sb.Append("]");

        return sb.ToString();
    }

    public T ToNumber()
    {
        if(_length != 1)
            throw new InvalidOperationException();
          
        return _data[0];
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
