using System.Numerics;
using System.Text;

namespace LinearAlgebra;

public class Vector<T> : IReadOnlyVector<T>
    where T : INumber<T>
{
    private readonly int _dimension;
    private T[] _data;

    public int Dimension => _dimension;

    public T LengthSquared
    {
        get 
        {
            T sum = T.Zero;
            for(int i = 0; i < _dimension; i++)
                sum += _data[i] * _data[i];

            return sum;
        }
    }

    public T this[int index]
    {
        get => _data[index];
        set => _data[index] = value;
    }

    public Vector(int length)
    {
        _dimension = length;
        _data = new T[length];
    }

    public Vector(T[] data)
    {
        _dimension = data.Length;
        _data = data;
    }

    public static Vector<T> Copy(IReadOnlyVector<T> other)
    {
        var data = new T[other.Dimension];
        for(int i = 0; i < other.Dimension; i++)
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
        if(lhs.Dimension != rhs.Dimension)
            throw new ArgumentException();
        
        T result = T.Zero;
        for(int i =0; i < lhs.Dimension; i++)
            result += lhs[i] * rhs[i];
        return result;
    }

    public void MultiplyByScalar(T scalar)
    {
        for(int i = 0; i < _dimension; i++)
            _data[i] *= scalar;
    }

    public override string ToString()
    {
        var sb = new StringBuilder();

        sb.Append("[");
        for(int i =0; i < _dimension; i++)
            sb.Append($"{_data[i]},\t");
        sb.Append("]");

        return sb.ToString();
    }

    public void SetValue(IReadOnlyVector<T> value)
    {
        if(_dimension != value.Dimension)
            throw new ArgumentException();
           
        _data = Copy(value)._data;
    }

    public void Substract(IReadOnlyVector<T> value)
    {
        if(value.Dimension != _dimension)
            throw new ArgumentException();

        for(int i = 0; i < _dimension; i++)
            _data[i] -= value[i];
    }

    public T ToNumber()
    {
        if(_dimension != 1)
            throw new InvalidOperationException();
          
        return _data[0];
    }

    public IReadOnlyMatrix<T> ToMatrixCached(int height, int width)
        => new Matrix<T>(height, width, _data);

    public static Vector<T> Sum(IReadOnlyVector<T> lhs, IReadOnlyVector<T> rhs)
    {
        if(lhs.Dimension != rhs.Dimension)
            throw new ArgumentException();
        
        return CreateFromFunction(lhs.Dimension, i => lhs[i] + rhs[i]);
    }

    public static Vector<T> Difference(IReadOnlyVector<T> lhs, IReadOnlyVector<T> rhs)
    {
        if(lhs.Dimension != rhs.Dimension)
            throw new ArgumentException();
        
        return CreateFromFunction(lhs.Dimension, i => lhs[i] - rhs[i]);
    }
}
