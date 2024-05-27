using System.Numerics;

namespace LinearAlgebra;

public class Vector<T>
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

    public void Add(Vector<T> vector)
    {
        for(int i = 0; i < _dimension; i++)
            _data[i] += vector[i];
    }

    public void SetZero()
    {
        for(int i = 0; i <_dimension; i++)
            _data[i] = T.Zero;
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

    public Matrix<T> AsHorizontalMatrix()
        => new Matrix<T>(1, _dimension, _data);
    
    public Matrix<T> AsMatrix(int height, int width)
        => new Matrix<T>(height, width, _data);

    public static void Addition(Vector<T> lhs, Vector<T> rhs, Vector<T> result)
    {
        for(int i = 0; i < result._data.Length; i++)
            result._data[i] = lhs._data[i] + rhs._data[i];
    }
    
    public static void Difference(Vector<T> lhs, Vector<T> rhs, Vector<T> result)
    {
        for(int i = 0; i < result._data.Length; i++)
            result._data[i] = lhs._data[i] - rhs._data[i];
    }

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
