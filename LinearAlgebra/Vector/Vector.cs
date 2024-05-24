using System.Numerics;
using System.Text;

namespace LinearAlgebra;

public class Vector<T> : IReadOnlyVector<T>
    where T : INumber<T>
{
    private readonly int _dimension;
    private readonly T[] _data;

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

    public Vector(T[] data)
    {
        _dimension = data.Length;
        _data = data;
    }

    public static Vector<T> CreateZeroVector(int length)
        => new Vector<T>(Enumerable.Range(0, length).Select(_ => T.Zero).ToArray());
    
    public static Vector<T> Create1DVector(T value)
        => new Vector<T>(new T[] {value});

    public void Add(IReadOnlyVector<T> vector)
    {
        for(int i = 0; i < _dimension; i++)
            _data[i] += vector[i];
    }

    public Vector<T> Scale(T value)
    {
        for(int i = 0; i < _dimension; i++)
            _data[i] *= value;
        
        return this;
    }
    
    public void CopyValuesFrom(Vector<T> value)
    {
        for(int i = 0; i < _data.Length; i++)
            _data[i] = value._data[i];
    }

    public T ToNumber()
    {
        if(_dimension != 1)
            throw new InvalidOperationException();
          
        return _data[0];
    }

    public Matrix<T> As1DMatrix()
        => new Matrix<T>(1, _dimension, _data);

    public static Vector<T> operator +(Vector<T> lhs, Vector<T> rhs)
    {
        var _data = new T[lhs.Dimension];
        for(int i = 0; i < _data.Length; i++)
            _data[i] = lhs[i] + rhs[i];

        return new Vector<T>(_data);
    }

    public static Vector<T> operator -(Vector<T> lhs, Vector<T> rhs)
    {
        var _data = new T[lhs.Dimension];
        for(int i = 0; i < _data.Length; i++)
            _data[i] = lhs[i] - rhs[i];

        return new Vector<T>(_data);
    }
}
